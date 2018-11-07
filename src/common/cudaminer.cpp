//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include <iostream>
#include <iomanip>
#include "../../include/cudaminer.h"
#include "../../argon2-gpu/include/argon2-cuda/processingunit.h"
#include "../../argon2-gpu/include/argon2-cuda/programcontext.h"
#include "../../argon2-gpu/include/argon2-cuda/device.h"
#include "../../argon2-gpu/include/argon2-cuda/globalcontext.h"
#include "../../argon2-gpu/include/argon2-cuda/cudaexception.h"

using namespace std;

static void setCudaDevice(int deviceIndex)
{
    int currentIndex = -1;
    argon2::cuda::CudaException::check(cudaGetDevice(&currentIndex));
    if (currentIndex != deviceIndex) {
        argon2::cuda::CudaException::check(cudaSetDevice(deviceIndex));
    }
}

void CudaMiningDevice::initialize(const Params & p)
{
    globalCtx.reset(
        new argon2::cuda::GlobalContext());
    
    auto &devices = globalCtx->getAllDevices();
    auto device = &devices[p.deviceIndex];

    setCudaDevice(p.deviceIndex);
    progCtx.reset(
        new argon2::cuda::ProgramContext(
            globalCtx.get(), { *device }, ARGON_TYPE, ARGON_VERSION));
}

// we MUST set device here
// when creating ProcessingUnit, cudaMalloc & cudaStreamCreate are called and they will operate on the current device

CudaMiner::CudaMiner(
    argon2::cuda::ProgramContext *progCtx, cudaStream_t & stream,
    argon2::MemConfig memoryConfig,
    Stats * pStats, MinerSettings & settings, Updater * pUpdater) :
    Miner(memoryConfig, pStats, settings, pUpdater),
    progCtx(progCtx),
    stream(stream)
{
    const auto INITIAL_BLOCK_TYPE = BLOCK_GPU;

    OptParams optPrms = configureArgon(
        AroConfig::passes(INITIAL_BLOCK_TYPE),
        AroConfig::memCost(INITIAL_BLOCK_TYPE),
        AroConfig::lanes(INITIAL_BLOCK_TYPE));
}

CudaMiner::CudaMiner(Stats *s, MinerSettings *ms, uint32_t bs, Updater *u, size_t *deviceIndex) : Miner(s, ms, bs, u) {
    
    global = new argon2::cuda::GlobalContext();
    auto &devices = global->getAllDevices();

    if (*deviceIndex >= devices.size()) {
        cout << endl << "!!! Warning !!! invalid device index: -d " << *deviceIndex <<", will use device 0 instead" << endl << endl;
        *deviceIndex = 0;
    }
   
    // we MUST set device here
    // when creating ProcessingUnit, cudaMalloc & cudaStreamCreate are called and they will operate on the current device
    device = &devices[*deviceIndex];
    setCudaDevice(device->getDeviceIndex());
    progCtx = new argon2::cuda::ProgramContext(global, {*device}, type, version);

     auto nLanesMax = AroConfig::lanes(BLOCK_GPU);
     try {
        bool bySegment = false;

        OptParams optPrms = configure(
            AroConfig::passes(BLOCK_GPU),
            AroConfig::memCost(BLOCK_GPU),
            nLanesMax,
            bs);

        unit = new argon2::cuda::ProcessingUnit(
            progCtx, params, device, 
            getInitialBatchSize(), bySegment, optPrms);
    }
    catch (const std::exception& e) {
        cout << "processing unit creation failed, " << e.what() << ", try to reduce -b / -t values" << endl;
        exit(1);
    }

    // allocate pinned memory for result buffers
    for (uint32_t i = 0; i < getInitialBatchSize(); i++) {
        cudaError_t status = cudaMallocHost((void**)&(resultBuffers[i]), nLanesMax * 1024 /*ARGON2_BLOCK_SIZE*/);
        if (status != cudaSuccess) {
            std::cout << "Error allocating pinned host memory" << std::endl;
            exit(1);
        }
    }
}

void CudaMiner::reconfigureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t newBatchSize) {
    if (!needReconfigure(t_cost, m_cost, lanes, newBatchSize))
        return;
    //printf("-- CudaMiner::reconfigureArgon %d,%d,%d\n", t_cost, m_cost, lanes);
    OptParams optPrms = configure(t_cost, m_cost, lanes, newBatchSize);
    unit->reconfigureArgon(params, batchSize, optPrms);
}

void CudaMiner::deviceUploadTaskDataAsync() {
    // upload to GPU
    size_t size = getCurrentBatchSize();
    for (size_t j = 0; j < size; ++j) {
        std::string data = bases.at(j);
        unit->setPassword(j, data.data(), data.length());
    }
}

void CudaMiner::deviceLaunchTaskAsync() {
    unit->beginProcessing();
}

void CudaMiner::deviceFetchTaskResultAsync() {
    size_t size = getCurrentBatchSize();
    for (size_t j = 0; j < size; ++j) {
        unit->fetchResultAsync(j, resultBuffers[j]);
    }
}

bool CudaMiner::deviceResultsReady() {
    return unit->streamOperationsComplete();
}

void CudaMiner::deviceWaitForResults() {
    unit->syncStream();
}

size_t CudaMiner::getMemoryUsage() const {
    return unit->getMemoryUsage();
}

size_t CudaMiner::getMemoryUsedPerBatch() const {
    return unit->getMemoryUsedPerBatch();
}

