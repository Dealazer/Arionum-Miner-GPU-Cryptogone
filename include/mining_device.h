#pragma once

#include "aro_tools.h"
#include "../../argon2/src/core.h"

#include <vector>
#include <memory>
#include <sstream>
#include <iomanip>
#include <iostream>

template<typename QUEUE, typename BUFFER>
class IMiningDevice {
public:
    virtual void initialize(uint32_t deviceIndex) = 0;
    virtual QUEUE* newQueue() = 0;
    virtual BUFFER* newBuffer(size_t size) = 0;
    virtual void writeBuffer(BUFFER* buf, const void* src, size_t size) = 0;
    virtual size_t maxAllocSize() const = 0;
    virtual bool testAlloc(size_t) = 0;
    virtual ~IMiningDevice() {};
};

template<typename QUEUE, typename BUFFER>
class MiningDevice : public IMiningDevice<QUEUE, BUFFER> {
public:
    QUEUE& queue(uint32_t i) {
        return *queues[i];
    }

protected:
    std::vector<std::unique_ptr<BUFFER>> buffers;
    std::vector<std::unique_ptr<QUEUE>> queues;
};

template<typename QUEUE, typename BUFFER>
class AroMiningDevice : public MiningDevice<QUEUE, BUFFER> {
public:
    const argon2::MemConfig& memConfig(uint32_t threadIndex) {
        return minersConfigs[threadIndex];
    }

    const std::string & buffersDesc() {
        return buffersDesc_;
    }

    friend class AroMiningDeviceFactory;

protected:
    std::string toGb(size_t nBytes) const {
        double gb = nBytes / 1073741824.0;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << gb << " GB";
        return oss.str();
    }

private:
    const bool USE_SINGLE_TASK_FOR_CPU_BLOCKS = true;
    std::vector<argon2::MemConfig> minersConfigs;

    struct AroMemoryInfo {
    public:
        AroMemoryInfo(uint32_t batchSizeGPU) {
            // compute GPU blocks mem cost
            auto tGPU = AroConfig::passes(BLOCK_GPU);
            auto mGPU = AroConfig::memCost(BLOCK_GPU);
            auto lGPU = AroConfig::lanes(BLOCK_GPU);
            argon2::Argon2Params paramsBlockGPU(
                32, "cifE2rK4nvmbVgQu", 16, nullptr, 0, nullptr, 0, tGPU, mGPU, lGPU);
            blocksPerHashGPU = paramsBlockGPU.getMemoryBlocks();
            memPerHashGPU = blocksPerHashGPU * argon2::ARGON2_BLOCK_SIZE;
            memPerTaskGPU = batchSizeGPU * memPerHashGPU;

            // compute CPU blocks mem cost
            auto tCPU = AroConfig::passes(BLOCK_CPU);
            auto mCPU = AroConfig::memCost(BLOCK_CPU);
            auto lCPU = AroConfig::lanes(BLOCK_CPU);
            argon2::Argon2Params paramsBlockCPU(
                32, "0KVwsNr6yT42uDX9", 16, nullptr, 0, nullptr, 0, tCPU, mCPU, lCPU);
            optPrmsCPU = precomputeArgon(&paramsBlockCPU);
            optPrmsCPU.mode = argon2::PRECOMPUTE_SHUFFLE;
            uint32_t blocksPerHashCPU = optPrmsCPU.customBlockCount;
            memPerHashCPU = blocksPerHashCPU * argon2::ARGON2_BLOCK_SIZE;
        }

        uint32_t blocksPerHashGPU;
        uint32_t blocksPerHashCPU;
        size_t memPerHashGPU;
        size_t memPerHashCPU;
        size_t memPerTaskGPU;
        argon2::OptParams optPrmsCPU;
    };

    void configureForAroMining(uint32_t nTasks, uint32_t batchSizeGPU) {
        // one queue per task
        for (uint32_t i = 0; i < nTasks; i++)
            this->newQueue();
        
        AroMemoryInfo aro(batchSizeGPU);

        // can we can allocate a single buffer for each task ?
        bool singleBufferPerGPUTask = this->testAlloc(aro.memPerTaskGPU);

        // compute blocks buffers sizes
        size_t totalMemPerTaskGPU = aro.memPerTaskGPU;
        size_t maxGPUBufferSize = 
            singleBufferPerGPUTask ? totalMemPerTaskGPU : this->maxAllocSize();
        std::vector<size_t> taskBuffersSizes;
        
        size_t avail = totalMemPerTaskGPU;
        while (avail > 0) {
            auto size = std::min(avail, maxGPUBufferSize);
            taskBuffersSizes.push_back(size);
            avail -= size;
        }
        while (taskBuffersSizes.size() > MAX_BLOCKS_BUFFERS)
            taskBuffersSizes.pop_back();

        size_t totalBlocksBufferMem = 0;
        std::vector<size_t> buffersSizes;
        for (uint32_t i = 0; i < nTasks; i++) {
            for (auto bufSize : taskBuffersSizes) {
                buffersSizes.push_back(bufSize);
                totalBlocksBufferMem += bufSize;
            }
        }

        // allocate block buffers
        std::vector<void*> buffers;
        for (auto it : buffersSizes)
            buffers.push_back(this->newBuffer(it));

        // save a description for showing to user
        std::ostringstream ossBuffersDesc;
        ossBuffersDesc << buffers.size() << " block buffer(s), ";
        ossBuffersDesc << toGb(totalBlocksBufferMem) << " used";
        buffersDesc_ = ossBuffersDesc.str();

        // allocate index buffer
        BUFFER * indexBuffer = nullptr;
        if (aro.optPrmsCPU.mode == argon2::PRECOMPUTE_LOCAL_STATE ||
            aro.optPrmsCPU.mode == argon2::PRECOMPUTE_SHUFFLE) {
            size_t indexSize = 
                aro.optPrmsCPU.customIndexNbSteps * sizeof(argon2_precomputed_index_t);
            indexBuffer = this->newBuffer(indexSize); // missing CL_MEM_READ_ONLY
            this->writeBuffer(indexBuffer, aro.optPrmsCPU.customIndex, indexSize);
        }

        // assign the buffers & set batch sizes
        minersConfigs.clear();
        size_t taskOffset = 0;
        for (uint32_t i = 0; i < nTasks; i++) {
            argon2::MemConfig mc;

            //std::cout << "--- Task " << i << std::endl;

            // set GPU blocks buffers for task
            for (size_t slot = 0; slot < taskBuffersSizes.size(); slot++) {
                auto & buf = buffers[taskOffset + slot];
                auto size = buffersSizes[taskOffset + slot];
                
                mc.blocksBuffers[BLOCK_GPU][slot] = buf;
                mc.batchSizes[BLOCK_GPU][slot] = size / aro.memPerHashGPU;

                //std::cout
                //    << "GPU Slot " << slot << " => " << toGb(size) 
                //    << " (" << mc.blocksBuffers[BLOCK_GPU][slot] << ")"
                //    << std::endl;
            }

            // set CPU blocks buffers for task
            if (USE_SINGLE_TASK_FOR_CPU_BLOCKS) {
                if (i == 0) {
                    size_t curSlot = 0;
                    size_t curBuffer = 0;
                    while (curBuffer < buffers.size() && curSlot < (size_t)MAX_BLOCKS_BUFFERS) {
                        auto batchSize = buffersSizes[curBuffer] / aro.memPerHashCPU;
                        if (batchSize == 0) {
                            curBuffer++;
                            continue;
                        }
                        mc.blocksBuffers[BLOCK_CPU][curSlot] = buffers[curBuffer];
                        mc.batchSizes[BLOCK_CPU][curSlot] = batchSize;

                        //std::cout
                        //    << "CPU Slot " << curSlot << " => " << toGb(buffersSizes[curBuffer])
                        //    << " (" << mc.blocksBuffers[BLOCK_CPU][curSlot] << ")"
                        //    << std::endl;

                        curSlot++;
                        curBuffer++;
                    }

                    mc.indexBuffer = indexBuffer;
                }
            }
            else {
                throw std::logic_error("USE_SINGLE_TASK_FOR_CPU_BLOCKS==false not implemented");
            }

            // set in & out buffers sizes
            uint32_t totalNonces = (uint32_t)std::max(
                mc.getTotalHashes(BLOCK_GPU),
                mc.getTotalHashes(BLOCK_CPU));
            const size_t MAX_LANES = 4;
            const size_t IN_BLOCKS_MAX_SIZE = MAX_LANES * 2 * argon2::ARGON2_BLOCK_SIZE;
            mc.in = IN_BLOCKS_MAX_SIZE * totalNonces;
            const size_t OUT_BLOCKS_MAX_SIZE = MAX_LANES * argon2::ARGON2_BLOCK_SIZE;
            mc.out = OUT_BLOCKS_MAX_SIZE * totalNonces;

            // save miner mem config
            minersConfigs.push_back(mc);

            taskOffset += taskBuffersSizes.size();
        }
    }

    std::string buffersDesc_;
};

class AroMiningDeviceFactory {
public:
    template<typename T>
    static T* create(
        std::uint32_t deviceIndex,
        std::uint32_t nTasks,
        std::uint32_t batchSizeGPU) {
        auto pDevice = new T();
        pDevice->initialize(deviceIndex);
        try {
            pDevice->configureForAroMining(nTasks, batchSizeGPU);
        }
        catch (const std::exception & e) {
            std::cout << "-- Device " 
                << std::setfill('0') << std::setw(2)  << deviceIndex
                << ": buffers creation failed, "
                << "miner probably uses too much GPU memory, "
                << "try to reduce -b / -t parameters"
                << std::endl;
            throw std::logic_error(e.what());
        }
        return pDevice;
    }
};
