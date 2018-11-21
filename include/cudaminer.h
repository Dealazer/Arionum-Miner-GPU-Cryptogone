//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_CUDAMINER_H
#define ARIONUM_GPU_MINER_CUDAMINER_H

#include "miner.h"
#include "updater.h"
#include "mining_device.h"

#include <argon2-cuda/globalcontext.h>
#include <argon2-cuda/processingunit.h>
#include <argon2-cuda/programcontext.h>
#include <argon2-cuda/device.h>

using QueueWrapper = argon2::cuda::QueueWrapper;
using BufferWrapper = argon2::cuda::BufferWrapper;
using MiningDeviceBase = AroMiningDevice<QueueWrapper, BufferWrapper>;

class CudaMiningDevice : public MiningDeviceBase
{
public:
    void initialize(uint32_t deviceIndex) override;
    QueueWrapper * newQueue() override;
    BufferWrapper * newBuffer(size_t size) override;
    void writeBuffer(BufferWrapper * buf, const void * str, size_t size) override;
    argon2::cuda::ProgramContext& programContext() { return *progCtx; };
    size_t maxAllocSize() const override;
    bool testAlloc(size_t) override;

private:
    std::unique_ptr<argon2::cuda::GlobalContext> globalCtx;
    std::unique_ptr<argon2::cuda::ProgramContext> progCtx;
};

class CudaMiner : public AroMiner {
public:
    CudaMiner(
       argon2::cuda::ProgramContext &, QueueWrapper &,
       argon2::MemConfig, const Services &,
       argon2::OPT_MODE);

   bool resultsReady() override;

protected:
    void reconfigureKernel() override;
    void uploadInputs_Async() override;
    void fetchResults_Async() override;
    void run_Async() override;
    uint8_t * resultsPtr() override;

private:
    std::unique_ptr<argon2::cuda::ProcessingUnit> unit;
};

#endif //ARIONUM_GPU_MINER_CUDAMINER_H
