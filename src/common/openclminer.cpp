//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include <iostream>
#include <iomanip>

#include "../../include/openclminer.h"

using namespace std;

OpenClMiner::OpenClMiner(Stats *s, MinerSettings *ms, uint32_t bs, Updater *u, size_t *deviceIndex)
        : Miner(s, ms, bs, u) {
    global = new argon2::opencl::GlobalContext();
    auto &devices = global->getAllDevices();
    device = &devices[*deviceIndex];
    progCtx = new argon2::opencl::ProgramContext(global, {*device}, type, version,
                                                 const_cast<char *>("./argon2-gpu/data/kernels/"));

    auto nLanes = 4;
    params = new argon2::Argon2Params(ARGON_OUTLEN, salt.data(), ARGON_SALTLEN, nullptr, 0, nullptr, 0, 4, 16384, 4);

    try {
        bool precompute = ms->precompute();
        unit = new argon2::opencl::ProcessingUnit(progCtx, params, device, getInitialBatchSize(), false, precompute);
    }
    catch (const std::exception& e) {
        cout << "Error: exception while creating opencl unit: " << e.what() << ", try to reduce batch size (-b parameter), exiting now :-(" << endl;
        exit(1);
    }

    for (uint32_t i = 0; i < getInitialBatchSize(); i++) {
        resultBuffers[i] = new uint8_t[nLanes * 1024 /*1024 is ARGON2_BLOCK_SIZE*/];
    }
}

void OpenClMiner::reconfigureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t newBatchSize) {
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
        unit->fetchResultAsync(j, resultBuffers[j]);
    }    
}

void OpenClMiner::deviceWaitForResults() {
    unit->waitForResults();
}

bool OpenClMiner::deviceResultsReady() {
    return unit->resultsReady();
}

size_t OpenClMiner::getMemoryUsage() const {
    return unit->getMemoryUsage();
}

size_t OpenClMiner::getMemoryUsedPerBatch() const {
    return unit->getMemoryUsedPerBatch();
}
