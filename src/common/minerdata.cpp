//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/minerdata.h"
#include <sstream>

const std::string BLOCK_TYPE_NAMES[3]{
    "Masternode",
    "GPU",
    "CPU"
};

const std::string BLOCK_TYPE_NAMES_SHORT[3]{
    "MN",
    "GPU",
    "CPU"
};

const std::string& blockTypeName(BLOCK_TYPE b) {
    return BLOCK_TYPE_NAMES[b];
}

const std::string& blockTypeNameShort(BLOCK_TYPE b) {
    return BLOCK_TYPE_NAMES_SHORT[b];
}

const string *MinerData::getStatus() const {
    return &status;
}

const string *MinerData::getDifficulty() const {
    return &difficulty;
}

const string *MinerData::getLimit() const {
    return &limit;
}

const string *MinerData::getBlock() const {
    return &block;
}

const string *MinerData::getPublic_key() const {
    return &public_key;
}

uint32_t MinerData::getHeight() const
{
    return height;
}

string MinerData::getArgonPrmsStr() const
{
    ostringstream oss;
    oss << argon_memory << "," << argon_threads << ", " << argon_time;
    return oss.str();
}

bool MinerData::isNewBlock(const string *newBlock) {
    return block != *newBlock;
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
    os << "type       : " << blockTypeName(data.getType()) << std::endl;
    os << "argon prms : " << data.getArgonPrmsStr() << std::endl;
    return os;
}

bool operator==(const MinerData &lhs, const MinerData &rhs) {
    return *lhs.getBlock() == *rhs.getBlock();
}

bool operator!=(const MinerData &lhs, const MinerData &rhs) {
    return !(lhs == rhs);
}

