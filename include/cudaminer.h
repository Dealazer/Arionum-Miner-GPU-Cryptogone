//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_CUDAMINER_H
#define ARIONUM_GPU_MINER_CUDAMINER_H

#include <argon2-cuda/globalcontext.h>
#include <argon2-cuda/processingunit.h>

#include "../argon2-gpu/include/argon2-cuda/programcontext.h"
#include "../argon2-gpu/include/argon2-cuda/device.h"

#include "miner.h"
#include "updater.h"

typedef argon2::MiningDeviceBase<
    argon2::cuda::ProgramContext,
    cudaStream_t,
    void*> DeviceBase;

class CudaMiningDevice : public DeviceBase
{
public:
    CudaMiningDevice(const Params &p) {
        initialize(p);
    }

    virtual void initialize(const Params &p);
private:
    std::unique_ptr<argon2::cuda::GlobalContext> globalCtx;
    std::unique_ptr<argon2::cuda::ProgramContext> progCtx;
};

class CudaMiner : public Miner {
public:
   CudaMiner(
       argon2::cuda::ProgramContext *, cudaStream_t &stream,
       argon2::MemConfig memConfig,
       Stats *pStats, MinerSettings &settings, Updater *pUpdater);
   
   void deviceUploadTaskDataAsync();
   void deviceLaunchTaskAsync();
   void deviceFetchTaskResultAsync();
   bool deviceResultsReady();
   void deviceWaitForResults();
   void reconfigureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t batchSize);
   size_t getMemoryUsage() const;
   size_t getMemoryUsedPerBatch() const;

private:
    argon2::cuda::ProgramContext * progCtx;
    cudaStream_t &stream;

    argon2::cuda::ProcessingUnit *unit;
    argon2::cuda::GlobalContext *global;
};

#endif //ARIONUM_GPU_MINER_CUDAMINER_H
