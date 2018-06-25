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

long MinerData::getLongDiff() const {
    return longDiff;
}

std::string strOrNull(const std::string* s) {
    if (s)
        return *s;
    else
        return string("null");
}

ostream &operator<<(ostream &os, const MinerData &data) {
    os << "status     : " << strOrNull(data.getStatus()) << std::endl;
    os << "difficulty : " << strOrNull(data.getDifficulty()) << std::endl;
    os << "block      : " << strOrNull(data.getBlock()) << std::endl;
    os << "limit      : " << strOrNull(data.getLimit()) << std::endl;
    os << "public_key : " << strOrNull(data.getPublic_key()) << std::endl;
    return os;
}

bool operator==(const MinerData &lhs, const MinerData &rhs) {
    return *lhs.getBlock() == *rhs.getBlock();
}

bool operator!=(const MinerData &lhs, const MinerData &rhs) {
    return !(lhs == rhs);
}

