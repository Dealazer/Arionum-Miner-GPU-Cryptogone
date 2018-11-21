#pragma once

#include <vector>
#include <memory>
#include "aro_tools.h"
#include "../../argon2/src/core.h"

template<typename QUEUE, typename BUFFER>
class IMiningDevice {
public:
    virtual void initialize(uint32_t deviceIndex) = 0;
    virtual QUEUE* newQueue() = 0;
    virtual BUFFER* newBuffer(size_t size) = 0;
    virtual void writeBuffer(BUFFER* buf, const void* src, size_t size) = 0;
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

    std::string toGb(size_t nBytes) {
        double gb = nBytes / 1073741824.0;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << gb << " GB";
        return oss.str();
    }

    void configureForAroMining(uint32_t nTasks, uint32_t batchSizeGPU) {
        // one queue per task
        for (uint32_t i = 0; i < nTasks; i++)
            this->newQueue();
        
        // compute blocks buffers sizes
        AroMemoryInfo aro(batchSizeGPU);
        std::vector<size_t> buffersSizes;
        for (uint32_t i = 0; i < nTasks; i++) {
            buffersSizes.push_back(aro.memPerTaskGPU);
        }

        // allocate block buffers
        std::vector<void*> buffers;
        for (auto it : buffersSizes)
            buffers.push_back(this->newBuffer(it));

        // save a description for showing to user
        std::ostringstream ossBuffersDesc;
        ossBuffersDesc << buffers.size() << " block buffers, ";
        size_t memUsed = 0;
        for (auto it : buffersSizes)
            memUsed += it;
        ossBuffersDesc << toGb(memUsed) << " used";
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
        for (uint32_t i = 0; i < nTasks; i++) {
            argon2::MemConfig mc;

            // set GPU blocks buffers for task
            mc.batchSizes[BLOCK_GPU][0] = batchSizeGPU;
            mc.blocksBuffers[BLOCK_GPU][0] = buffers[i];

            // set CPU blocks buffers for task
            if (USE_SINGLE_TASK_FOR_CPU_BLOCKS) {
                if (i == 0) {
                    for (uint32_t j = 0; j < nTasks; j++) {
                        mc.batchSizes[BLOCK_CPU][j] = aro.memPerTaskGPU / aro.memPerHashCPU;
                        mc.blocksBuffers[BLOCK_CPU][j] = buffers[j];
                    }
                    mc.indexBuffer = indexBuffer;
                }
            }
            else {
                mc.batchSizes[BLOCK_CPU][0] = aro.memPerTaskGPU / aro.memPerHashCPU;
                mc.blocksBuffers[BLOCK_CPU][0] = buffers[i];
                mc.indexBuffer = indexBuffer;
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
        pDevice->configureForAroMining(nTasks, batchSizeGPU);
        return pDevice;
    }
};
