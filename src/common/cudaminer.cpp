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

CudaMiner::CudaMiner(Stats *s, MinerSettings *ms, uint32_t bs, Updater *u, size_t *deviceIndex) : Miner(s, ms, bs, u) {
    global = new argon2::cuda::GlobalContext();
    auto &devices = global->getAllDevices();

    if (*deviceIndex >= devices.size()) {
        cout << endl << "!!! Warning !!! invalid device index: -d " << *deviceIndex <<", will use device 0 instead" << endl << endl;
        *deviceIndex = 0;
    }

    device = &devices[*deviceIndex];

    // we MUST set device here
    // when creating ProcessingUnit, cudaMalloc & cudaStreamCreate are called and they will operate on the current device
    setCudaDevice(device->getDeviceIndex());

    progCtx = new argon2::cuda::ProgramContext(global, {*device}, type, version);
    
    auto nLanes = 4;
    params = new argon2::Argon2Params(32, salt.data(), 16, nullptr, 0, nullptr, 0, 4, 16384, nLanes);

    const auto OPTIMIZATION_MODE = BASELINE;
    try {
        unit = new argon2::cuda::ProcessingUnit(progCtx, params, device, getInitialBatchSize(), OPTIMIZATION_MODE);
    }
    catch (const std::exception& e) {
        cout << "processing unit creation failed, " << e.what() << ", try to reduce -b / -t values" << endl;
        exit(1);
    }
    
    for (uint32_t i = 0; i < getInitialBatchSize(); i++) {
        cudaError_t status = cudaMallocHost((void**)&(resultBuffers[i]), nLanes * 1024 /*ARGON2_BLOCK_SIZE*/);
        if (status != cudaSuccess) {
            std::cout << "Error allocating pinned host memory" << std::endl;
            exit(1);
        }
    }
}

void CudaMiner::reconfigureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t newBatchSize) {
    if (params->getTimeCost() == t_cost &&
        params->getMemoryCost() == m_cost &&
        params->getLanes() == lanes)
    {
        return;
    }

    batchSize = newBatchSize;
    if (params) {
        delete params;
    }
    params = new argon2::Argon2Params(ARGON_OUTLEN, salt.data(), ARGON_SALTLEN, nullptr, 0, nullptr, 0, t_cost, m_cost, lanes);

    unit->reconfigureArgon(params, batchSize);
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

void CudaMiner::deviceWaitForResults()
{
    unit->syncStream();
}

