//
// Created by guli on 31/01/18.
//
#include "../../include/minerdata.h"


string *MinerData::getStatus() const {
    return status;
}

string *MinerData::getDifficulty() const {
    return difficulty;
}

string *MinerData::getLimit() const {
    return limit;
}

string *MinerData::getBlock() const {
    return block;
}

string *MinerData::getPublic_key() const {
    return public_key;
}

bool MinerData::isNewBlock(string *newBlock) {
    return *block != *newBlock;
}


ostream &operator<<(ostream &os, const MinerData &data) {
    os << " difficulty: " << *data.getDifficulty()
       << " - limit: " << *data.getLimit()
       << " - block: " << *data.getBlock()
       << " - public_key: " << *data.getPublic_key()
       << endl;
    return os;
}

bool operator==(const MinerData &lhs, const MinerData &rhs) {
    return *lhs.getBlock() == *rhs.getBlock();
}

bool operator!=(const MinerData &lhs, const MinerData &rhs) {
    return !(lhs == rhs);
}

