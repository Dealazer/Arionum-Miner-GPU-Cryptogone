//
// Created by guli on 31/01/18.
//

#include <iostream>
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

CudaMiner::CudaMiner(Stats *s, MinerSettings *ms, Updater *u, size_t *deviceIndex) : Miner(s, ms, u) {
    global = new argon2::cuda::GlobalContext();
    auto &devices = global->getAllDevices();

    if (*deviceIndex >= devices.size()) {
        cout << endl << "!!! Warning !!! invalid device index: -d " << *deviceIndex <<", will use device 0 instead" << endl << endl;
        *deviceIndex = 0;
    }

    device = &devices[*deviceIndex];
    cout << "using device " << *deviceIndex << " - " << device->getName() << endl;
    cout << "using salt " << salt << endl;

    // we MUST set device here
    // when creating ProcessingUnit, cudaMalloc & cudaStreamCreate are called and they will operate on the current device
    setCudaDevice(device->getDeviceIndex());

    progCtx = new argon2::cuda::ProgramContext(global, {*device}, type, version);
    params = new argon2::Argon2Params(32, salt.data(), 16, nullptr, 0, nullptr, 0, 1, 524288, 1);

    try {
        unit = new argon2::cuda::ProcessingUnit(progCtx, params, device, *settings->getBatchSize(), false, false);
    }
    catch (const std::exception& e) {
        cout << "Error: exception while creating cudaminer unit: " << e.what() << ", try to reduce batch size (-b parameter), exiting now :-(" << endl;
        exit(1);
    }
    
    for (int i = 0; i < *settings->getBatchSize(); i++) {
        cudaError_t status = cudaMallocHost((void**)&(resultBuffers[i]), 1024 /*ARGON2_BLOCK_SIZE*/);
        if (status != cudaSuccess) {
            std::cout << "Error allocating pinned host memory" << std::endl;
            exit(1);
        }
    }
}

#include <chrono>
using std::chrono::high_resolution_clock;

struct TTimer {
    void start();
    void end(float &tgt);
    std::chrono::time_point<std::chrono::high_resolution_clock> startT, endT;
};

void TTimer::start()
{
    startT = high_resolution_clock::now();
}

void TTimer::end(float &tgt)
{
    endT = high_resolution_clock::now();
    std::chrono::duration<float> duration = endT - startT;
    tgt = duration.count();
    assert(tgt >= 0.f);
}

void CudaMiner::deviceUploadTaskDataAsync() {
    // upload to GPU
    size_t size = *settings->getBatchSize();
    for (size_t j = 0; j < size; ++j) {
        std::string data = bases.at(j);
        unit->setPassword(j, data.data(), data.length());
    }
}

void CudaMiner::deviceLaunchTaskAsync() {
    unit->beginProcessing();
}

void CudaMiner::deviceFetchTaskResultAsync() {
    size_t size = *settings->getBatchSize();
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

