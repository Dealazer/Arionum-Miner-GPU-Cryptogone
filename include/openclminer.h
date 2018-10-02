//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_OPENCLMINER_H
#define ARIONUM_GPU_MINER_OPENCLMINER_H

#include "miner.h"
#include "updater.h"

#include <argon2-gpu-common/argon2-common.h>
#include <argon2-opencl/programcontext.h>
#include <argon2-gpu-common/argon2params.h>
#include <argon2-opencl/processingunit.h>

#include <vector>

class OpenClMiningDevice {
public:
    OpenClMiningDevice(
        size_t deviceIndex,
        uint32_t nTasks, uint32_t batchSizeGPU);

    argon2::MemConfig getMemConfig(int taskId);

    argon2::opencl::ProgramContext *getProgramContext() {
        return progCtx;
    }

    cl::CommandQueue& getQueue(int index) {
        return queues[index];
    }

protected:
    cl::Buffer indexBuffer;
    vector<cl::Buffer> buffers;
    vector<argon2::MemConfig> minersConfigs;
    argon2::opencl::ProgramContext *progCtx;
    vector<cl::CommandQueue> queues;
};

class OpenClMiner : public Miner {
public:
    OpenClMiner(
        argon2::opencl::ProgramContext *, cl::CommandQueue &queue,
        argon2::MemConfig memConfig,
        Stats *pStats, MinerSettings &settings, Updater *pUpdater);

    void reconfigureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes);
    
    void deviceUploadTaskDataAsync();
    void deviceLaunchTaskAsync();
    void deviceFetchTaskResultAsync();
    void deviceWaitForResults();
    bool deviceResultsReady();

private:
    argon2::opencl::ProcessingUnit *unit;
    argon2::opencl::ProgramContext *progCtx;
    cl::CommandQueue& queue;
};

#endif //ARIONUM_GPU_MINER_OPENCLMINER_H
