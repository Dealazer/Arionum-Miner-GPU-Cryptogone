//
// Created by guli on 31/01/18.
//

#ifndef ARIONUM_GPU_MINER_OPENCLMINER_H
#define ARIONUM_GPU_MINER_OPENCLMINER_H

#include "miner.h"

class OpenClMiner : public Miner {
public:
    void mine();

    explicit OpenClMiner(int *id);
};

#endif //ARIONUM_GPU_MINER_OPENCLMINER_H
