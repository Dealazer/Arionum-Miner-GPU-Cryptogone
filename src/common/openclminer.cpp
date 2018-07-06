//
// Created by guli on 31/01/18.
//

#include <iostream>
#include "../../include/openclminer.h"

using namespace std;

OpenClMiner::OpenClMiner(Stats *s, MinerSettings *ms, Updater *u, size_t *deviceIndex)
        : Miner(s, ms, u) {
    global = new argon2::opencl::GlobalContext();
    auto &devices = global->getAllDevices();
    device = &devices[*deviceIndex];
    cout << "using device #" << *deviceIndex << " - " << device->getName() << endl;
    cout << "using salt " << salt << endl;
    progCtx = new argon2::opencl::ProgramContext(global, {*device}, type, version,
                                                 const_cast<char *>("./argon2-gpu/data/kernels/"));
    params = new argon2::Argon2Params(32, salt.data(), 16, nullptr, 0, nullptr, 0, 1, 524288, 1);

    try {
        unit = new argon2::opencl::ProcessingUnit(progCtx, params, device, *settings->getBatchSize(), false, false);
    }
    catch (const std::exception& e) {
        cout << "Error: exception while creating opencl unit: " << e.what() << ", try to reduce batch size (-b parameter), exiting now :-(" << endl;
        exit(1);
    }

    for (int i = 0; i < *settings->getBatchSize(); i++) {
        resultBuffers[i] = new uint8_t[1024 /*ARGON2_BLOCK_SIZE*/];
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
