//
// Created by guli on 31/01/18.
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
    os << "uniqid: " << *settings.getUniqid()
       << " - pool address: " << *settings.getPoolAddress()
       << " - private key: " << *settings.getPrivateKey();
    return os;
}

