#include "../../include/mining_device.h"
#include "../../include/perfscope.h"
#include "../../argon2/src/core.h"

#include <map>
#include <algorithm>

const bool USE_SINGLE_TASK_FOR_CPU_BLOCKS = true;

argon2::OptParams precomputeArgon(argon2::Argon2Params * params) {

    static std::map<uint32_t, argon2::OptParams> s_precomputeCache;

    auto m_cost = params->getMemoryCost();
    std::map<uint32_t, argon2::OptParams>::const_iterator it =
        s_precomputeCache.find(m_cost);
    if (it == s_precomputeCache.end()) {
        PERFSCOPE("INDEX PRECOMPUTE");
        argon2_instance_t inst;
        memset(&inst, 0, sizeof(inst));
        inst.context_ptr = nullptr;
        inst.lanes = params->getLanes();
        inst.segment_length = params->getSegmentBlocks();
        inst.lane_length = inst.segment_length * ARGON2_SYNC_POINTS;
        inst.memory = nullptr;
        inst.memory_blocks = params->getMemoryBlocks();
        inst.passes = params->getTimeCost();
        inst.threads = params->getLanes();
        inst.type = Argon2_i;

        auto nSteps = argon2i_index_size(&inst);
        const uint32_t* pIndex = (uint32_t*)(new argon2_precomputed_index_t[nSteps]);
        uint32_t blockCount = argon2i_precompute(&inst, (argon2_precomputed_index_t*)pIndex);

        argon2::OptParams prms;
        prms.customBlockCount = blockCount;
        prms.customIndex = pIndex;
        prms.customIndexNbSteps = nSteps;
        s_precomputeCache[m_cost] = prms;
    }

    return s_precomputeCache[m_cost];
}

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

std::vector<argon2::MemConfig> AroMiningDeviceFactory::configureForAroMining(
    IMiningDevice& device,
    uint32_t nTasks, uint32_t batchSizeGPU) {

    AroMemoryInfo aro(batchSizeGPU);

    std::vector<void*> buffers;
    for (uint32_t i = 0; i < nTasks; i++) {
        device.newQueue();
        buffers.push_back(device.newBuffer(aro.memPerTaskGPU));
    }

    void* indexBuffer = nullptr;
    if (aro.optPrmsCPU.mode == argon2::PRECOMPUTE_LOCAL_STATE ||
        aro.optPrmsCPU.mode == argon2::PRECOMPUTE_SHUFFLE) {
        size_t indexSize = aro.optPrmsCPU.customIndexNbSteps * 3 * sizeof(uint32_t);
        indexBuffer = device.newBuffer(indexSize); // missing CL_MEM_READ_ONLY
        device.writeBuffer(indexBuffer, aro.optPrmsCPU.customIndex, indexSize);
    }

    std::vector<argon2::MemConfig> configs;
    for (uint32_t i = 0; i < nTasks; i++) {
        argon2::MemConfig mc;
        mc.batchSizes[BLOCK_GPU][0] = batchSizeGPU;
        mc.blocksBuffers[BLOCK_GPU][0] = buffers[i];

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

        uint32_t totalNonces = (uint32_t)std::max(
            mc.getTotalHashes(BLOCK_GPU),
            mc.getTotalHashes(BLOCK_CPU));

        const size_t MAX_LANES = 4;
        const size_t IN_BLOCKS_MAX_SIZE = MAX_LANES * 2 * ARGON2_BLOCK_SIZE;
        mc.in = IN_BLOCKS_MAX_SIZE * totalNonces;

        const size_t OUT_BLOCKS_MAX_SIZE = MAX_LANES * ARGON2_BLOCK_SIZE;
        mc.out = OUT_BLOCKS_MAX_SIZE * totalNonces;

        configs.push_back(mc);
    }

    return configs;
}
