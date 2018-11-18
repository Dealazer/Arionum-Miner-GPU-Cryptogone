//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINERSETTINGS_H
#define ARIONUM_GPU_MINER_MINERSETTINGS_H

#include <cstring>
#include <iostream>
#include <mutex>

#include "minerdata.h"

class MinerSettings {
public:
    MinerSettings(
        const std::string & poolAddress, 
        const std::string & privateKey, 
        const std::string & uniqid, 
        bool mineGPU, bool mineCPU, bool showLastHashrate) :
        poolAddress_(poolAddress),
        privateKey_(privateKey),
        uniqid_(uniqid),
        mineGpuBlocks(mineGPU),
        mineCpuBlocks(mineCPU),
        showLastHashrate(showLastHashrate) {
    };

    friend std::ostream &operator<<(std::ostream &os, const MinerSettings &settings);

    const std::string & poolAddress() const;
    const std::string & privateKey() const;
    const std::string & uniqueID() const;
    bool canMineBlock(BLOCK_TYPE type) const;
    bool useLastHashrateInsteadOfRoundAvg() const { return showLastHashrate; };

private:
    const std::string &poolAddress_;
    const std::string &privateKey_;
    const std::string &uniqid_;
    bool mineGpuBlocks;
    bool mineCpuBlocks;
    bool showLastHashrate;
};

#endif //ARIONUM_GPU_MINER_MINERSETTINGS_H
