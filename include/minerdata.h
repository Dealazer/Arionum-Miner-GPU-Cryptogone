//
// Created by guli on 31/01/18.
//

#ifndef ARIONUM_GPU_MINER_MINERDATA_H
#define ARIONUM_GPU_MINER_MINERDATA_H

#include <cstring>
#include <iostream>
#include <mutex>
#include <atomic>

using namespace std;

class MinerData {
private:
    string *status = new string("");
    string *difficulty = new string("56648645");
    std::atomic<int> limit;
    string *block = new string("");
    string *public_key = new string("");
    std::mutex mutex;

    MinerData(const MinerData &data);

public:
    MinerData() : limit(100000) {};

    void updateData(string s, string d, size_t l, string b, string p);

    MinerData *getCopy();

    string *getStatus() const;

    string *getDifficulty() const;

    int getLimit() const;

    string *getBlock() const;

    string *getPublic_key() const;

    bool isNewBlock(string *newBloack);

    friend ostream &operator<<(ostream &os, const MinerData &data);

    friend bool operator==(const MinerData &lhs, const MinerData &rhs);

    friend bool operator!=(const MinerData &lhs, const MinerData &rhs);

};

#endif //ARIONUM_GPU_MINER_MINERDATA_H
