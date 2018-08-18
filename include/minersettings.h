//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINERSETTINGS_H
#define ARIONUM_GPU_MINER_MINERSETTINGS_H

#include <cstring>
#include <iostream>
#include <mutex>

using namespace std;

class MinerSettings {
private:
    string *poolAddress;
    string *privateKey;
    string *uniqid;
public:
    size_t *batchSize;

public:
    MinerSettings(string *pa, string *pk, string *ui, size_t *bs) : poolAddress(pa),
                                                        privateKey(pk),
                                                        uniqid(ui),
                                                        batchSize(bs) {};

    friend ostream &operator<<(ostream &os, const MinerSettings &settings);

    string *getPoolAddress() const;

    string *getPrivateKey() const;

    string *getUniqid() const;

    size_t *getBatchSize() const;
};

#endif //ARIONUM_GPU_MINER_MINERSETTINGS_H
