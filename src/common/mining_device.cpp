#include "../../include/mining_device.h"
#include "../../include/perfscope.h"
#include "../../argon2/src/core.h"

#include <map>
#include <algorithm>
#include <cstring> // memset

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
