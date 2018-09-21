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

OpenClMiner::OpenClMiner(
    size_t deviceIndex, size_t maxMem, 
    Stats *pStats, MinerSettings &settings, Updater *pUpdater) : 
    Miner(maxMem, pStats, settings, pUpdater) {

    global = new argon2::opencl::GlobalContext();

    auto &devices = global->getAllDevices();
    device = &devices[deviceIndex];

#if USE_PROGRAM_CACHE
    cl_device_id deviceID = device->getCLDevice()();
    auto it = s_programCache.find(deviceID);
    if (it == s_programCache.end()) {
        s_programCache.insert(
            std::make_pair(
                deviceID,
                new argon2::opencl::ProgramContext(
                    global, { *device }, type, version,
                    "./argon2-gpu/data/kernels/")));
    }
    progCtx = s_programCache[deviceID];
#else
    progCtx = new argon2::opencl::ProgramContext(
        global, {*device}, type, version,
        "./argon2-gpu/data/kernels/");
#endif
}

#define VERBOSE_CONFIGURE (1)

argon2::MemConfig OpenClMiner::configure(size_t maxMemUsage) {
    argon2::MemConfig mc;

    // get mem per hash needed for CPU round
    auto optParamsCPU = configureArgon(
        Miner::getPasses(BLOCK_CPU),
        Miner::getMemCost(BLOCK_CPU),
        Miner::getLanes(BLOCK_CPU));
    uint32_t cpuBlocksPerHash = (optParamsCPU.mode == PRECOMPUTE) ?
        optParamsCPU.customBlockCount :
        params->getMemoryBlocks();
    size_t memPerHashCPU = cpuBlocksPerHash * ARGON2_BLOCK_SIZE;

    // get mem per hash needed for GPU round
    auto optParamsGPU = configureArgon(
        Miner::getPasses(BLOCK_GPU),
        Miner::getMemCost(BLOCK_GPU),
        Miner::getLanes(BLOCK_GPU));
    uint32_t gpuBlocksPerHash = params->getMemoryBlocks();
    size_t memPerHashGPU = gpuBlocksPerHash * ARGON2_BLOCK_SIZE;

    // get mem needed for precomputed indices
    mc.index = 0;
    if (optParamsCPU.mode == PRECOMPUTE) {
        mc.index = optParamsCPU.customIndexNbSteps * 3 * sizeof(cl_uint);
    }

    // get mem needed for sending nonces
    const size_t MAX_LANES = 4;
    const size_t IN_BLOCKS_MAX_SIZE = MAX_LANES * 2 * ARGON2_BLOCK_SIZE;
    size_t maxTotalNonces = maxMemUsage / std::min(memPerHashCPU, memPerHashGPU);
    mc.in = IN_BLOCKS_MAX_SIZE * maxTotalNonces;

    // get mem needed for getting back results
    const size_t OUT_BLOCKS_MAX_SIZE = MAX_LANES * ARGON2_BLOCK_SIZE;
    mc.out = OUT_BLOCKS_MAX_SIZE * maxTotalNonces;

    // !! to review !!
    // get mem needed for warp shuffle
    // !! to review !!
    size_t warpShuffleSize = 32 * MAX_LANES * sizeof(cl_uint) * 2;
    
    // get device mem info
    auto cl_dev = device->getCLDevice();
    size_t deviceTotalMem = cl_dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    size_t deviceMaxAlloc = cl_dev.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

#if VERBOSE_CONFIGURE
    cout << "maxMemUsage    = " << maxMemUsage << endl;
    cout << "deviceTotalMem = " << deviceTotalMem << endl;
    cout << "deviceMaxAlloc = " << deviceMaxAlloc << endl;
    cout << "memPerHashCPU  = " << memPerHashCPU << endl;
    cout << "memPerHashGPU  = " << memPerHashGPU << endl;
    cout << "index          = " << mc.index << endl;
    cout << "in             = " << mc.in << endl;
    cout << "out            = " << mc.out << endl;
    cout << "warpShuffle    = " << warpShuffleSize << endl;
#endif

    size_t totalAvail = std::min(maxMemUsage, deviceTotalMem);
    size_t blocksMaxSize = totalAvail - (mc.index + mc.in + mc.out + warpShuffleSize);

#if VERBOSE_CONFIGURE
    double efficiency = ((double)blocksMaxSize / (double)totalAvail);
    cout << "blocksMaxSize  = " << blocksMaxSize
        << std::fixed << std::setprecision(2)
        << " (" << (100.0 * efficiency) << "%)" << endl;
#endif

    // buffer 0
    size_t nHashesCPU = std::min(blocksMaxSize, deviceMaxAlloc) / memPerHashCPU;
    size_t buf0Size = nHashesCPU * memPerHashCPU;
    if (nHashesCPU == 0) {
        cout << "Error: device does not have enough memory to compute a hash !" << endl;
        exit(1);
    }
    size_t nHashesGPU = buf0Size / memPerHashGPU;
    
#if VERBOSE_CONFIGURE
    double usagePercent = 100.0 * 
        ((double)(nHashesGPU * memPerHashGPU) / (double)buf0Size);
    cout << "buffer 0" << endl;
    cout << "  nHashesCPU = " << nHashesCPU << endl;
    cout 
        << "  nHashesGPU = " << nHashesGPU 
        << " (" << usagePercent << "%)"
        << endl;
    cout << "  size       = " << buf0Size << endl;
#endif

    mc.blocksBuffers[0] = (uint32_t)buf0Size;
    mc.batchSizes[BLOCK_CPU][0] = nHashesCPU;
    mc.batchSizes[BLOCK_GPU][0] = nHashesGPU;

    // other buffers
    for (int i = 1; i < MAX_BLOCKS_BUFFERS; i++) {
        mc.blocksBuffers[i] = 0;
        mc.batchSizes[BLOCK_CPU][i] = 0;
        mc.batchSizes[BLOCK_GPU][i] = 0;
    }

    return mc;
}

bool OpenClMiner::createUnit() {
    const auto INITIAL_BLOCK_TYPE = BLOCK_CPU;
    
    t_optParams optPrms = configureArgon(
        Miner::getPasses(INITIAL_BLOCK_TYPE),
        Miner::getMemCost(INITIAL_BLOCK_TYPE),
        Miner::getLanes(INITIAL_BLOCK_TYPE));

    unit = new argon2::opencl::ProcessingUnit(
        device,
        progCtx,
        params,
        memConfig,
        optPrms,
        INITIAL_BLOCK_TYPE);

    return true;
}

void OpenClMiner::reconfigureArgon(
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes) {

    if (!needReconfigure(t_cost, m_cost, lanes))
        return;

    t_optParams optPrms =
        configureArgon(t_cost, m_cost, lanes);

    unit->reconfigureArgon(
        device,
        params,
        optPrms,
        (t_cost == 1) ? BLOCK_CPU : BLOCK_GPU);
}

void OpenClMiner::deviceUploadTaskDataAsync() {
    unit->uploadInputDataAsync(bases);
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
    }
    return queueFinished;
}

