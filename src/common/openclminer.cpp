//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include <iostream>
#include <iomanip>

#include "../../include/openclminer.h"

using namespace std;

void OpenClMiner::printInfo() {
    auto batchSize = *settings->getBatchSize();
    cout << "Device       : " << device->getName() << endl;
    cout << "Batch size   : " << batchSize << endl;
    cout << "VRAM usage   : " << std::fixed << std::setprecision(2) << 
        (float)(batchSize * params->getMemorySize()) / (1024.f*1024.f*1024.f) << " GB" << endl;
    cout << "Salt         : " << salt << endl;
}

OpenClMiner::OpenClMiner(Stats *s, MinerSettings *ms, Updater *u, size_t *deviceIndex)
        : Miner(s, ms, u) {
    global = new argon2::opencl::GlobalContext();
    auto &devices = global->getAllDevices();
    device = &devices[*deviceIndex];
    progCtx = new argon2::opencl::ProgramContext(global, {*device}, type, version,
                                                 const_cast<char *>("./argon2-gpu/data/kernels/"));

    auto nLanes = 4;
    params = new argon2::Argon2Params(32, salt.data(), 16, nullptr, 0, nullptr, 0, 4, 16384, nLanes);

    try {
        unit = new argon2::opencl::ProcessingUnit(progCtx, params, device, *settings->getBatchSize(), false, false);
    }
    catch (const std::exception& e) {
        cout << "Error: exception while creating opencl unit: " << e.what() << ", try to reduce batch size (-b parameter), exiting now :-(" << endl;
        exit(1);
    }

    for (int i = 0; i < *settings->getBatchSize(); i++) {
        resultBuffers[i] = new uint8_t[nLanes * 1024 /*ARGON2_BLOCK_SIZE*/];
    }
}

void OpenClMiner::deviceUploadTaskDataAsync() {
    size_t size = *settings->getBatchSize();
    for (size_t j = 0; j < size; ++j) {
        std::string data = bases.at(j);
        unit->setPassword(j, data.data(), data.length());
    }
}

void OpenClMiner::deviceLaunchTaskAsync() {
    unit->runKernelAsync();
}

void OpenClMiner::deviceFetchTaskResultAsync() {
    size_t size = *settings->getBatchSize();
    for (size_t j = 0; j < size; ++j) {
        unit->fetchResultAsync(j, resultBuffers[j]);
    }    
}

void OpenClMiner::deviceWaitForResults() {
    unit->waitForResults();
}

bool OpenClMiner::deviceResultsReady() {
    return unit->resultsReady();
}
