//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/minersettings.h"

const std::string & MinerSettings::poolAddress() const {
    return poolAddress_;
}

const std::string & MinerSettings::privateKey() const {
    return privateKey_;
}

const std::string & MinerSettings::uniqueID() const {
    return uniqid_;
}

std::ostream &operator<<(std::ostream &os, const MinerSettings &settings) {
	os << "worker id     : " << settings.uniqueID() << std::endl
       << "pool address  : " << settings.poolAddress() << std::endl
       << "wallet address: " << settings.privateKey();
    return os;
}

bool MinerSettings::canMineBlock(BLOCK_TYPE type) const {
    return
        (type == BLOCK_CPU && mineCpuBlocks) ||
        (type == BLOCK_GPU && mineGpuBlocks);
}

