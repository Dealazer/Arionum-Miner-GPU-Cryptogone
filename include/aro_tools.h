#pragma once

#include "block_type.h"

#include <argon2-gpu-common/argon2-common.h>
#include <argon2-gpu-common/argon2params.h>

struct AroConfig {
    static uint32_t memCost(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 524288 : 16384;
    }

    static uint32_t passes(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 1 : 4;
    }

    static uint32_t lanes(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 1 : 4;
    }
};

argon2::OptParams precomputeArgon(argon2::Argon2Params * params);

