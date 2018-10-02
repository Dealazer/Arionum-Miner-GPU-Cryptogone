//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include <iostream>
#include <iomanip>

#include "../../include/openclminer.h"

using namespace argon2;
using namespace std;
using argon2::t_optParams;
using argon2::PRECOMPUTE;
using argon2::BASELINE;

#define USE_PROGRAM_CACHE (1)
#if USE_PROGRAM_CACHE
#include <map>
static std::map<cl_device_id, argon2::opencl::ProgramContext*> s_programCache;
#endif

namespace cl {
    bool s_logCLErrors = true;
}

std::string toGB(size_t size) {
    double GB = (double)size / (1024.f * 1024.f * 1024.f);
    ostringstream os;
    os << std::fixed << std::setprecision(3) << GB << " GB";
    return os.str();
}

OpenClMiner::OpenClMiner(
    argon2::opencl::ProgramContext *progCtx, argon2::MemConfig memoryConfig,
    Stats *pStats, MinerSettings &settings, Updater *pUpdater) :
    Miner(memoryConfig, pStats, settings, pUpdater),
    progCtx(progCtx),
    queue(progCtx->getContext())
{
    const auto INITIAL_BLOCK_TYPE = BLOCK_GPU;

    t_optParams optPrms = configureArgon(
        Miner::getPasses(INITIAL_BLOCK_TYPE),
        Miner::getMemCost(INITIAL_BLOCK_TYPE),
        Miner::getLanes(INITIAL_BLOCK_TYPE));

    unit = new argon2::opencl::ProcessingUnit(
                queue,
                progCtx,
                params,
                memConfig,
                optPrms,
                INITIAL_BLOCK_TYPE);
}

void OpenClMiner::reconfigureArgon(
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes) {
    if (!needReconfigure(t_cost, m_cost, lanes))
        return;

    t_optParams optPrms =
        configureArgon(t_cost, m_cost, lanes);

    unit->reconfigureArgon(params, optPrms, (t_cost == 1) ? BLOCK_CPU : BLOCK_GPU);
}

void OpenClMiner::deviceUploadTaskDataAsync() {
#if (!OPEN_CL_SKIP_MEM_TRANSFERS)
    unit->uploadInputDataAsync(bases);
#endif
}

void OpenClMiner::deviceLaunchTaskAsync() {
    unit->runKernelAsync();
}

void OpenClMiner::deviceFetchTaskResultAsync() {
    unit->fetchResultsAsync();
}

void OpenClMiner::deviceWaitForResults() {
    unit->waitForResults();
}

bool OpenClMiner::deviceResultsReady() {
    bool queueFinished = unit->resultsReady();
    if (queueFinished) {
#if (!OPEN_CL_SKIP_MEM_TRANSFERS)
        auto blockType = (params->getLanes() == 1) ? 
            BLOCK_CPU : BLOCK_GPU;
        int curHash = 0;
        for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
            auto nHashes = memConfig.batchSizes[blockType][i];
            for (auto j = 0; j < nHashes; j++) {
                resultsPtrs[i][j] = unit->getResultPtr(curHash);
                curHash++;
            }
        }
#endif
    }
    return queueFinished;
}

OpenClMiningDevice::OpenClMiningDevice(
    size_t deviceIndex,
    uint32_t nTasks, uint32_t batchSizeGPU)
{
    // only support up to 4 buffers for now
    if (nTasks > MAX_BLOCKS_BUFFERS) {
        std::ostringstream oss;
        oss << "-t value must be <= " << nTasks;
        throw std::logic_error(oss.str());
    }

    // compute GPU blocks mem cost
    auto tGPU = Miner::getPasses(BLOCK_GPU);
    auto mGPU = Miner::getMemCost(BLOCK_GPU);
    auto lGPU = Miner::getLanes(BLOCK_GPU);
    argon2::Argon2Params paramsBlockGPU(
        32, "cifE2rK4nvmbVgQu", 16, nullptr, 0, nullptr, 0, tGPU, mGPU, lGPU);
    uint32_t blocksPerHashGPU = paramsBlockGPU.getMemoryBlocks();
    size_t memPerHashGPU = blocksPerHashGPU * ARGON2_BLOCK_SIZE;
    size_t memPerTaskGPU = batchSizeGPU * memPerHashGPU;

    // compute CPU blocks mem cost
    auto tCPU = Miner::getPasses(BLOCK_CPU);
    auto mCPU = Miner::getMemCost(BLOCK_CPU);
    auto lCPU = Miner::getLanes(BLOCK_CPU);
    argon2::Argon2Params paramsBlockCPU(
        32, "0KVwsNr6yT42uDX9", 16, nullptr, 0, nullptr, 0, tCPU, mCPU, lCPU);
    t_optParams optPrmsCPU = Miner::precomputeArgon(&paramsBlockCPU);
    optPrmsCPU.mode = PRECOMPUTE;
    uint32_t blocksPerHashCPU = optPrmsCPU.customBlockCount;
    size_t memPerHashCPU = blocksPerHashCPU * ARGON2_BLOCK_SIZE;

    // create context
    argon2::opencl::GlobalContext global;
    auto &devices = global.getAllDevices();
    auto device = &devices[deviceIndex];

#if USE_PROGRAM_CACHE
    cl_device_id deviceID = device->getCLDevice()();
    auto it = s_programCache.find(deviceID);
    if (it == s_programCache.end()) {
        s_programCache.insert(
            std::make_pair(
                deviceID,
                new argon2::opencl::ProgramContext(
                    &global, { *device }, ARGON_TYPE, ARGON_VERSION,
                    "./argon2-gpu/data/kernels/")));
    }
    progCtx = s_programCache[deviceID];
#else
    progCtx = new argon2::opencl::ProgramContext(
        &global, { *device }, type, version,
        "./argon2-gpu/data/kernels/");
#endif
    auto context = progCtx->getContext();

    // utility to create a buffer
    auto allocBuffer = [&](
        size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE) -> cl::Buffer {
        // allocate buffer
        cl::Buffer buf(context, flags, size);
        // warm it up
        cl::CommandQueue queue(context, device->getCLDevice(), 0);
        uint8_t dummy = 0xFF;
        queue.enqueueWriteBuffer(buf, true, size-1, 1, &dummy);
        return buf;
    };

    // create nTasks block buffers
    for (uint32_t i = 0; i < nTasks; i++) {
        buffers.push_back(allocBuffer(memPerTaskGPU));
    }

    // create index buffer
    size_t indexSize = 0;
    if (optPrmsCPU.mode == PRECOMPUTE) {
        indexSize = optPrmsCPU.customIndexNbSteps * 3 * sizeof(cl_uint);
        indexBuffer = allocBuffer(indexSize, CL_MEM_READ_ONLY);
        cl::CommandQueue queue(context, device->getCLDevice(), 0);
        queue.enqueueWriteBuffer(
            indexBuffer, true, 0, indexSize, optPrmsCPU.customIndex);
    }

    // create mem configs for GPU tasks
    for (uint32_t i = 0; i < nTasks; i++) {
        MemConfig mc;
        mc.batchSizes[BLOCK_GPU][0] = batchSizeGPU;
        mc.blocksBuffers[BLOCK_GPU][0] = &buffers[i];

        if (i == 0) {
            for (uint32_t j = 0; j < nTasks; j++) {
                mc.batchSizes[BLOCK_CPU][j] = memPerTaskGPU / memPerHashCPU;
                mc.blocksBuffers[BLOCK_CPU][j] = &buffers[j];
            }
            mc.indexBuffer = &indexBuffer;
        }

        uint32_t totalNonces = (uint32_t)std::max(
            mc.getTotalHashes(BLOCK_GPU),
            mc.getTotalHashes(BLOCK_CPU));

        const size_t MAX_LANES = 4;
        const size_t IN_BLOCKS_MAX_SIZE = MAX_LANES * 2 * ARGON2_BLOCK_SIZE;
        mc.in = IN_BLOCKS_MAX_SIZE * totalNonces;

        const size_t OUT_BLOCKS_MAX_SIZE = MAX_LANES * ARGON2_BLOCK_SIZE;
        mc.out = OUT_BLOCKS_MAX_SIZE * totalNonces;

        minersConfigs.push_back(mc);
    }

    auto pBuffer = (cl::Buffer*)minersConfigs[0].blocksBuffers[BLOCK_CPU][0];
}

argon2::MemConfig OpenClMiningDevice::getMemConfig(int taskId) {
    return minersConfigs[taskId];
}
