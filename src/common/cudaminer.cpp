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

#include "argon2.h"
#include "../../argon2/src/core.h"

#include <map>

using namespace std;

static void setCudaDevice(int deviceIndex)
{	
    int currentIndex = -1;
    argon2::cuda::CudaException::check(cudaGetDevice(&currentIndex));
    if (currentIndex != deviceIndex) {
        argon2::cuda::CudaException::check(cudaSetDevice(deviceIndex));
    }
}

t_optParams CudaMiner::configure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t bs) {
    batchSize = bs;

    if (params)
        delete params;
    params = new argon2::Argon2Params(32, salt.data(), 16, nullptr, 0, nullptr, 0, t_cost, m_cost, lanes);

    t_optParams optPrms;
    memset(&optPrms, 0, sizeof(optPrms));

#if 1
    optPrms.mode = (lanes == 1 && t_cost == 1) ? PRECOMPUTE : BASELINE;
#else
    optPrms.mode = BASELINE;
#endif

    if (optPrms.mode == PRECOMPUTE) {
        
        static std::map<uint32_t, t_optParams> s_precomputeCache;

        auto &it = s_precomputeCache.find(m_cost);
        if (it == s_precomputeCache.end()) {
            argon2_instance_t inst;
            memset(&inst, 0, sizeof(inst));
            inst.context_ptr = nullptr;
            inst.lanes = params->getLanes();
            inst.segment_length = params->getSegmentBlocks();
            inst.lane_length = inst.segment_length * ARGON2_SYNC_POINTS;
            inst.memory = nullptr;
            inst.memory_blocks = params->getMemoryBlocks();
            inst.passes = params->getTimeCost();
            inst.threads = params->getLanes();
            inst.type = Argon2_i;

            auto nSteps = argon2i_index_size(&inst);
            uint32_t* pIndex = (uint32_t*)(new argon2_precomputed_index_t[nSteps]);
            uint32_t blockCount = argon2i_precompute(&inst, (argon2_precomputed_index_t*)pIndex);

            optPrms.customBlockCount = blockCount;
            optPrms.customIndex = pIndex;
            optPrms.customIndexNbSteps = nSteps;

            s_precomputeCache[m_cost] = optPrms;
        }
        optPrms = s_precomputeCache[m_cost];
    }
    return optPrms;
}

CudaMiner::CudaMiner(Stats *s, MinerSettings *ms, uint32_t bs, Updater *u, size_t *deviceIndex) : Miner(s, ms, bs, u) {
    global = new argon2::cuda::GlobalContext();
    auto &devices = global->getAllDevices();

    if (*deviceIndex >= devices.size()) {
        cout << endl << "!!! Warning !!! invalid device index: -d " << *deviceIndex <<", will use device 0 instead" << endl << endl;
        *deviceIndex = 0;
    }

    device = &devices[*deviceIndex];
    auto nLanes = 4;
    t_optParams optPrms = configure(4, 16384, nLanes, bs);
    
    // we MUST set device here
    // when creating ProcessingUnit, cudaMalloc & cudaStreamCreate are called and they will operate on the current device
    setCudaDevice(device->getDeviceIndex());
    progCtx = new argon2::cuda::ProgramContext(global, {*device}, type, version);
    
     try {
        bool bySegment = false;
        unit = new argon2::cuda::ProcessingUnit(progCtx, params, device, getInitialBatchSize(), bySegment, optPrms);
    }
    catch (const std::exception& e) {
        cout << "processing unit creation failed, " << e.what() << ", try to reduce -b / -t values" << endl;
        exit(1);
    }

    // allocate pinned memory for result buffers
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
        params->getLanes() == lanes &&
        newBatchSize == batchSize)
    {
        return;
    }
    
    //printf("-- CudaMiner::reconfigureArgon %d,%d,%d\n", t_cost, m_cost, lanes);

    t_optParams optPrms = configure(t_cost, m_cost, lanes, newBatchSize);
    if (this->batchSize != newBatchSize) {
        printf("problem changing batch size in reconfigureArgon\n");
        exit(1);
    }

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


