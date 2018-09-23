//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/minerdata.h"
#include <sstream>

using std::string;
using std::ostringstream;
using std::ostream;

const string BLOCK_TYPE_NAMES[3]{
    "GPU",
    "CPU",
    "Masternode"
};

const string BLOCK_TYPE_NAMES_SHORT[3]{
    "GPU",
    "CPU",
    "MN"
};

const string& blockTypeName(BLOCK_TYPE b) {
    return BLOCK_TYPE_NAMES[b];
}

const string& blockTypeNameShort(BLOCK_TYPE b) {
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
    oss << argon_threads << ", " << argon_memory << ", " << argon_time;
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
#if 1    
    os << "height     : " << data.getHeight() << std::endl;
    os << "type       : " << blockTypeName(data.getBlockType()) << std::endl;
    os << "difficulty : " << strOrNull(data.getDifficulty()) << std::endl;
    os << "argon2i    : " << data.getArgonPrmsStr() << std::endl;
#else
    os << "height     : " << data.getHeight() << std::endl;
    os << "type       : " << blockTypeName(data.getBlockType()) << std::endl;
    os << "difficulty : " << strOrNull(data.getDifficulty()) << std::endl;
    os << "argon2i    : " << data.getArgonPrmsStr() << std::endl;
    os << "limit      : " << strOrNull(data.getLimit()) << std::endl;
    os << "block      : " << strOrNull(data.getBlock()) << std::endl;
    os << "public_key : " << strOrNull(data.getPublic_key()) << std::endl;
    os << "status     : " << strOrNull(data.getStatus()) << std::endl;
#endif
    return os;
}

bool operator==(const MinerData &lhs, const MinerData &rhs) {
    return *lhs.getBlock() == *rhs.getBlock();
}

bool operator!=(const MinerData &lhs, const MinerData &rhs) {
    return !(lhs == rhs);
}

