//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include "../../include/cudaminer.h"
#include "../../argon2-gpu/include/argon2-cuda/processingunit.h"
#include "../../argon2-gpu/include/argon2-cuda/programcontext.h"
#include "../../argon2-gpu/include/argon2-cuda/device.h"
#include "../../argon2-gpu/include/argon2-cuda/globalcontext.h"
#include "../../argon2-gpu/include/argon2-cuda/cudaexception.h"

#include <cuda_runtime.h>
#include <argon2-cuda/cudaexception.h>

#include <iostream>
#include <iomanip>
#include <memory>

void CudaMiningDevice::initialize(uint32_t deviceIndex) {
    globalCtx.reset(new argon2::cuda::GlobalContext());
    auto & deviceRef = globalCtx->getAllDevices()[deviceIndex];
    progCtx.reset(new argon2::cuda::ProgramContext(
        deviceRef, ARGON_TYPE, ARGON_VERSION));
}

QueueWrapper * CudaMiningDevice::newQueue() {
    progCtx->getDevice().setAsCurrent();
    queues.emplace_back(new QueueWrapper());
    argon2::cuda::CudaException::check(cudaGetLastError());
    return queues.back().get();
}

BufferWrapper * CudaMiningDevice::newBuffer(size_t size) {
    progCtx->getDevice().setAsCurrent();
    buffers.emplace_back(new BufferWrapper(size));
    argon2::cuda::CudaException::check(cudaGetLastError());
    return buffers.back().get();
}

size_t CudaMiningDevice::maxAllocSize() const {
    progCtx->getDevice().setAsCurrent();
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

bool CudaMiningDevice::testAlloc(size_t size) {
    return size < maxAllocSize();
}

void CudaMiningDevice::writeBuffer(BufferWrapper * bufferWrapper, const void * str, size_t size) {
    argon2::cuda::CudaException::check(
        cudaMemcpy(bufferWrapper->buf, str, size, cudaMemcpyHostToDevice));
}

CudaMiner::CudaMiner(
    argon2::cuda::ProgramContext & progCtx, QueueWrapper & queue,
    argon2::MemConfig memConfig, const Services & services,
    argon2::OPT_MODE cpuBlocksOptimizationMode) :
    AroMiner(memConfig, services, cpuBlocksOptimizationMode),
    unit{} {
    using ProcessingUnit = argon2::cuda::ProcessingUnit;
    using MiningContext = argon2::cuda::KernelRunner::MiningContext;
    unit.reset(new ProcessingUnit(
        miningConfig, MiningContext { queue.stream, progCtx }));
    argon2::cuda::CudaException::check(cudaGetLastError());
}

bool CudaMiner::resultsReady() {
    return unit->streamOperationsComplete();
}

void CudaMiner::reconfigureKernel() {
    unit->configure();
}

void CudaMiner::uploadInputs_Async() {
    unit->uploadInputDataAsync(nonces.bases);
}

void CudaMiner::fetchResults_Async() {
    unit->fetchResultsAsync();
}

void CudaMiner::run_Async() {
    unit->beginProcessing();
}

uint8_t * CudaMiner::resultsPtr() {
    return unit->results();
}
