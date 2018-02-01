//
// Created by guli on 31/01/18.
//

#ifndef ARIONUM_GPU_MINER_CUDAMINER_H
#define ARIONUM_GPU_MINER_CUDAMINER_H

#include <argon2-cuda/globalcontext.h>
#include <argon2-cuda/processingunit.h>
#include "miner.h"
#include "updater.h"

#include "../argon2-gpu/include/argon2-cuda/processingunit.h"
#include "../argon2-gpu/include/argon2-cuda/programcontext.h"
#include "../argon2-gpu/include/argon2-cuda/device.h"
#include "../argon2-gpu/include/argon2-cuda/globalcontext.h"

class CudaMiner : public Miner {
private:
    argon2::cuda::ProcessingUnit *unit;
    argon2::cuda::ProgramContext *progCtx;
    argon2::cuda::GlobalContext *global;
    const argon2::cuda::Device *device;

public:
    void computeHash();

    explicit CudaMiner(Stats *s, MinerSettings *ms, Updater *u, size_t *deviceIndex);
};

#endif //ARIONUM_GPU_MINER_CUDAMINER_H
