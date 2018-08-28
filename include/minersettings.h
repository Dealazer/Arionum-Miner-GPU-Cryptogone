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
    bool precomputeRefs;

public:
    MinerSettings(string *pa, string *pk, string *ui, size_t *bs, bool precompute) : 
        poolAddress(pa),
        privateKey(pk),
        uniqid(ui),
        precomputeRefs(precompute) {
    };

    friend ostream &operator<<(ostream &os, const MinerSettings &settings);

    string *getPoolAddress() const;

    string *getPrivateKey() const;

    string *getUniqid() const;

    bool precompute() const;
};

#endif //ARIONUM_GPU_MINER_MINERSETTINGS_H
