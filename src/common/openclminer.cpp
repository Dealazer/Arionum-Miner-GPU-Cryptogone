//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include <iostream>
#include <iomanip>

#include "../../include/openclminer.h"

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

    auto nLanesMax = Miner::getLanes(BLOCK_GPU);
    try {
        bool bySegment = false;
        t_optParams optPrms = configureArgon(
            Miner::getPasses(BLOCK_GPU),
            Miner::getMemCost(BLOCK_GPU),
            nLanesMax);
        unit = new argon2::opencl::ProcessingUnit(
            progCtx, params, device, 
            getInitialBatchSize(), bySegment, optPrms);
    }
    catch (const std::exception& e) {
        cout << "Error: exception while creating opencl unit: " 
             << e.what() << ", try to reduce batch size (-b parameter), exiting now :-(" 
             << endl;
        exit(1);
    }
}

Miner::MemConfig OpenClMiner::configure(size_t maxMemUsage) {

    // get n blocks needed for CPU round
    auto optParamsCPU = configureArgon(
        Miner::getPasses(BLOCK_CPU),
        Miner::getMemCost(BLOCK_CPU),
        Miner::getLanes(BLOCK_CPU));
    uint32_t cpuBlocksPerHash = (optParamsCPU.mode == PRECOMPUTE) ?
        optParamsCPU.customBlockCount :
        params->getMemoryBlocks();

    // get n blocks needed for GPU round
    auto optParamsGPU = configureArgon(
        Miner::getPasses(BLOCK_GPU),
        Miner::getMemCost(BLOCK_GPU),
        Miner::getLanes(BLOCK_GPU));
    uint32_t gpuBlocksPerHash = params->getMemoryBlocks();

    //auto nBlocks = std::max(cpuBlocksPerHash, gpuBlocksPerHash);

    MemConfig mc;
    //mc.batchSizes;
    //mc.blocksBuffers;
    //mc.in;
    //mc.index;
    //mc.out;
}

bool OpenClMiner::initialize(MemConfig &mcfg) {
    return true;
}

void OpenClMiner::reconfigureArgon(
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t newBatchSize) {
    if (!needReconfigure(t_cost, m_cost, lanes, newBatchSize))
        return;
    t_optParams optPrms = configure(t_cost, m_cost, lanes, newBatchSize);
    unit->reconfigureArgon(params, batchSize, optPrms);
}

void OpenClMiner::deviceUploadTaskDataAsync() {
    size_t size = getCurrentBatchSize();
    for (size_t j = 0; j < size; ++j) {
        std::string data = bases.at(j);
        unit->setPassword(j, data.data(), data.length());
    }
}

void OpenClMiner::deviceLaunchTaskAsync() {
    unit->runKernelAsync();
}

void OpenClMiner::deviceFetchTaskResultAsync() {
    size_t size = getCurrentBatchSize();
    for (size_t j = 0; j < size; ++j) {
        unit->fetchResultAsync(j);
    }    
}

void OpenClMiner::deviceWaitForResults() {
    unit->waitForResults();
}

bool OpenClMiner::deviceResultsReady() {
    bool queueFinished = unit->resultsReady();
    if (queueFinished) {
        for (uint32_t i = 0; i < unit->getBatchSize(); i++) {
            resultBuffers[i] = unit->getResultPtr(i);
        }
    }
    return queueFinished;
}

//size_t OpenClMiner::getMemoryUsage() const {
//    return unit->getMemoryUsage();
//}

//size_t OpenClMiner::getMemoryUsedPerBatch() const {
//    return unit->getMemoryUsedPerBatch();
//}
