//
// Created by guli on 31/01/18.
//

#ifndef ARIONUM_GPU_MINER_MINER_H
#define ARIONUM_GPU_MINER_MINER_H

class Miner {
protected:
    int *id;
public:
    virtual void mine()=0;

    explicit Miner(int *id) : id(id){};

};

#endif //ARIONUM_GPU_MINER_MINER_H
