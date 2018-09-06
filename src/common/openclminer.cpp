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

OpenClMiner::OpenClMiner(Stats *s, MinerSettings *ms, uint32_t bs, Updater *u, size_t *deviceIndex)
        : Miner(s, ms, bs, u) {
    global = new argon2::opencl::GlobalContext();

    auto &devices = global->getAllDevices();
    device = &devices[*deviceIndex];

    progCtx = new argon2::opencl::ProgramContext(
        global, {*device}, type, version,
        "./argon2-gpu/data/kernels/");

    auto nLanesMax = 4;
    try {
        bool bySegment = false;
        t_optParams optPrms = configure(4, 16384, nLanesMax, bs);
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

size_t OpenClMiner::getMemoryUsage() const {
    return unit->getMemoryUsage();
}

size_t OpenClMiner::getMemoryUsedPerBatch() const {
    return unit->getMemoryUsedPerBatch();
}
