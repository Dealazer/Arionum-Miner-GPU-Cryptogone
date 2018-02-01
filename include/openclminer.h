//
// Created by guli on 31/01/18.
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
    void computeHash();

    OpenClMiner(Stats *s, MinerSettings *ms, Updater *pUpdater, size_t *deviceIndex);
};

#endif //ARIONUM_GPU_MINER_OPENCLMINER_H
