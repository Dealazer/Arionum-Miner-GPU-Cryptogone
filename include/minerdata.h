//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINERDATA_H
#define ARIONUM_GPU_MINER_MINERDATA_H

#include <cstring>
#include <iostream>
#include <mutex>
#include <atomic>
#include <string>

using namespace std;

enum BLOCK_TYPE {
    BLOCK_MASTERNODE,
    BLOCK_GPU,
    BLOCK_CPU
};

const std::string& blockTypeName(BLOCK_TYPE b);

class MinerData {
private:
    string status = "";
    string difficulty = "";
    string limit = "";
    string block = "";
    string public_key = "";
    long longDiff = 1;
    uint32_t height = 0;
    BLOCK_TYPE type = BLOCK_CPU;

public:
    uint32_t argon_memory = 0;
    uint32_t argon_threads = 0;
    uint32_t argon_time = 0;

    bool isValid() const {
        return height != 0;
    }

public:
    MinerData() = default;

    MinerData(
        string s,
        string d,
        string l,
        string b,
        string p,
        uint32_t _height,
        uint32_t _argon_memory,
        uint32_t _argon_threads,
        uint32_t _argon_time,
        BLOCK_TYPE _type)
    {
        status = s;
        difficulty = d;
        block = b;
        public_key = p;
        limit = l;
        longDiff = std::stol(d);

        argon_memory = _argon_memory;
        argon_threads = _argon_threads;
        argon_time = _argon_time;
        type = _type;
        height = _height;
    };

    const string *getStatus() const;

    const string *getDifficulty() const;

    const string *getLimit() const;

    const string *getBlock() const;

    const string *getPublic_key() const;

    uint32_t getHeight() const;

    BLOCK_TYPE getType() const { return type; };
    string getArgonPrmsStr() const;

    bool isNewBlock(const string *newBlock);

    friend ostream &operator<<(ostream &os, const MinerData &data);

    friend bool operator==(const MinerData &lhs, const MinerData &rhs);

    friend bool operator!=(const MinerData &lhs, const MinerData &rhs);

    long getLongDiff() const;
};

#endif //ARIONUM_GPU_MINER_MINERDATA_H
