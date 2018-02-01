//
// Created by guli on 31/01/18.
//
#include "../../include/minerdata.h"

MinerData::MinerData(const MinerData &data) {
    status = new string;
    *status = *data.status;
    difficulty = new string;
    *difficulty = *data.difficulty;
    limit += data.limit;
    block = new string;
    *block = *data.block;
    public_key = new string;
    *public_key = *data.public_key;
}

void MinerData::updateData(string s, string d, int *l, string b, string p) {
    std::lock_guard<std::mutex> lg(mutex);
    status->assign(s);
    difficulty->assign(d);
    limit = *l;
    block->assign(b);
    public_key->assign(p);
}

MinerData *MinerData::getCopy() {
    std::lock_guard<std::mutex> lg(mutex);
    return new MinerData(*this);
}

string *MinerData::getStatus() const {
    return status;
}

string *MinerData::getDifficulty() const {
    return difficulty;
}

int MinerData::getLimit() const {
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
       << " - limit: " << data.getLimit()
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

