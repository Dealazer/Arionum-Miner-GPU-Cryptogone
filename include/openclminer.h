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

class OpenClMiner : public Miner {
private:
    argon2::opencl::ProcessingUnit *unit;
    argon2::opencl::ProgramContext *progCtx;
    argon2::opencl::GlobalContext *global;
    const argon2::opencl::Device *device;

public:

    void deviceUploadTaskDataAsync();
    void deviceLaunchTaskAsync();
    void deviceFetchTaskResultAsync();
    void deviceWaitForResults();
    bool deviceResultsReady();
    void reconfigureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t batchSize);

    OpenClMiner(Stats *s, MinerSettings *ms, uint32_t bs, Updater *pUpdater, size_t *deviceIndex);
};

#endif //ARIONUM_GPU_MINER_OPENCLMINER_H
