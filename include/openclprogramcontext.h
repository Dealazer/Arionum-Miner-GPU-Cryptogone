//
// Created by guli on 01/02/18.
//

#ifndef ARIONUM_GPU_MINER_OPENCLPROGRAMCONTEXT_H
#define ARIONUM_GPU_MINER_OPENCLPROGRAMCONTEXT_H

#include "openclminer.h"

class OpenClProgramContext : public argon2::opencl::ProgramContext {

public:
    OpenClProgramContext(const argon2::opencl::GlobalContext *globalContext,
                         const vector<argon2::opencl::Device> &devices, argon2::Type type, argon2::Version version)
            : ProgramContext(globalContext, devices, type, version) {
        devices.reserve(devices.size());
        for (auto &device : devices) {
            this->devices.push_back(device.getCLDevice());
        }
        context = cl::Context(this->devices);

        program = KernelLoader::loadArgon2Program(
                // FIXME path:
                context, "./argon2-gpu/data/kernels", type, version);

    }

};

#endif //ARIONUM_GPU_MINER_OPENCLPROGRAMCONTEXT_H
