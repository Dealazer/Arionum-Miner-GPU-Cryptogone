//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_OPENCLMINER_H
#define ARIONUM_GPU_MINER_OPENCLMINER_H

#include "miner.h"
#include "updater.h"
#include "mining_device.h"

#include <argon2-gpu-common/argon2-common.h>
#include <argon2-gpu-common/argon2params.h>

#include <argon2-opencl/programcontext.h>
#include <argon2-opencl/processingunit.h>
#include <argon2-opencl/kernelrunner.h>

#include <vector>

typedef AroMiningDevice<cl::CommandQueue, cl::Buffer> MiningDeviceBase;

class OpenClMiningDevice : public MiningDeviceBase
{
public:
    void initialize(uint32_t deviceIndex) override;
    cl::CommandQueue* newQueue() override;
    cl::Buffer* newBuffer(size_t size) override;
    void writeBuffer(cl::Buffer * buf, const void * str, size_t size) override;
    argon2::opencl::ProgramContext* programContext() { return progCtx; }

private:
    argon2::opencl::GlobalContext globalCtx;
    argon2::opencl::ProgramContext* progCtx;
    argon2::opencl::Device device;
};

class OpenClMiner : public AroMiner {
public:
    OpenClMiner(
        argon2::opencl::ProgramContext *progCtx, cl::CommandQueue & refQueue,
        const argon2::MemConfig &memConfig, const Services& services,
        argon2::OPT_MODE gpuOptimizationMode);

    bool resultsReady() override;
    
protected:
    void reconfigureKernel() override;
    void uploadInputs_Async() override;
    void run_Async() override;
    void fetchResults_Async() override;
    uint8_t * resultsPtr() override;

private:
    argon2::opencl::KernelRunner::MiningContext ctx;
    std::unique_ptr<argon2::opencl::ProcessingUnit> pUnit;
};

#endif //ARIONUM_GPU_MINER_OPENCLMINER_H
