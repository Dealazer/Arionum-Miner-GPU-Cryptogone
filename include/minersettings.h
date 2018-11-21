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
        const std::string & stats_nodeUrl,
        const std::string & stats_nodeKey,
        bool mineGPU, bool mineCPU, bool showLastHashrate) :
        poolAddress_(poolAddress),
        privateKey_(privateKey),
        uniqid_(uniqid),
        stats_nodeUrl_(stats_nodeUrl),
        stats_nodeKey_(stats_nodeKey),
        mineGpuBlocks(mineGPU),
        mineCpuBlocks(mineCPU),
        showLastHashrate(showLastHashrate) {
    }

    friend std::ostream &operator<<(std::ostream &os, const MinerSettings &settings);

    const std::string & poolAddress() const;
    const std::string & privateKey() const;
    const std::string & uniqueID() const;
    bool canMineBlock(BLOCK_TYPE type) const;
    bool useLastHashrateInsteadOfRoundAvg() const { return showLastHashrate; };
    bool hasStatsNode() const { return stats_nodeUrl_.size() > 0; };
    std::string statsAPIUrl() const {
        if (!hasStatsNode())
            return{};
        return stats_nodeUrl_;
    }
    const std::string statsToken() const {
        return stats_nodeKey_;
    }
private:
    const std::string &poolAddress_;
    const std::string &privateKey_;
    const std::string &uniqid_;
    const std::string &stats_nodeUrl_;
    const std::string &stats_nodeKey_;
    bool mineGpuBlocks;
    bool mineCpuBlocks;
    bool showLastHashrate;
};

#endif //ARIONUM_GPU_MINER_MINERSETTINGS_H
