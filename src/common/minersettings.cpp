//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/minersettings.h"

string *MinerSettings::getPoolAddress() const {
    return poolAddress;
}

string *MinerSettings::getPrivateKey() const {
    return privateKey;
}

string *MinerSettings::getUniqid() const {
    return uniqid;
}

size_t *MinerSettings::getBatchSize() const {
    return batchSize;
}

ostream &operator<<(ostream &os, const MinerSettings &settings) {
	os << "uniqid        : " << settings.getUniqid()->c_str() << std::endl
       << "pool address  : " << settings.getPoolAddress()->c_str() << std::endl
       << "wallet address: " << settings.getPrivateKey()->c_str();
    return os;
}

