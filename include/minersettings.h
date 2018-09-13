//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINERSETTINGS_H
#define ARIONUM_GPU_MINER_MINERSETTINGS_H

#include <cstring>
#include <iostream>
#include <mutex>

#include "minerdata.h"

using namespace std;

class MinerSettings {
private:
    string *poolAddress;
    string *privateKey;
    string *uniqid;
    bool mineGpuBlocks;
    bool mineCpuBlocks;
    bool showLastHashrate;
public:
    MinerSettings(
        string *pa, string *pk, string *ui, size_t *bs,
        bool mineGPU, bool mineCPU, bool showLastHashrate) :
        poolAddress(pa),
        privateKey(pk),
        uniqid(ui),
        mineGpuBlocks(mineGPU),
        mineCpuBlocks(mineCPU),
        showLastHashrate(showLastHashrate) {
    };

    friend ostream &operator<<(ostream &os, const MinerSettings &settings);

    string *getPoolAddress() const;

    string *getPrivateKey() const;

    string *getUniqid() const;

    bool mineBlock(BLOCK_TYPE type) const;
    bool useLastHashrateInsteadOfRoundAvg() const { return showLastHashrate; };
};

#endif //ARIONUM_GPU_MINER_MINERSETTINGS_H
