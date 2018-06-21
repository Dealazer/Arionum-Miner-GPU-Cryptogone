//
// Created by guli on 31/01/18.
//

#ifndef ARIONUM_GPU_MINER_MINERDATA_H
#define ARIONUM_GPU_MINER_MINERDATA_H

#include <cstring>
#include <iostream>
#include <mutex>
#include <atomic>
#include <string>

using namespace std;

class MinerData {
private:
    string *status = new string("");
    string *difficulty = new string("56648645");
    string *limit = new string("");
    string *block = new string("");
    string *public_key = new string("");
    long longDiff = 1;

public:
    MinerData(string s, string d, string l, string b, string p) {
        status = new string(s);
        difficulty = new string(d);
        block = new string(b);
        public_key = new string(p);
        limit = new string(l);
        longDiff = std::stol(d);
    };

    string *getStatus() const;

    string *getDifficulty() const;

    string *getLimit() const;

    string *getBlock() const;

    string *getPublic_key() const;

    bool isNewBlock(string *newBlock);

    friend ostream &operator<<(ostream &os, const MinerData &data);

    friend bool operator==(const MinerData &lhs, const MinerData &rhs);

    friend bool operator!=(const MinerData &lhs, const MinerData &rhs);

    long getLongDiff() const;

};

#endif //ARIONUM_GPU_MINER_MINERDATA_H
