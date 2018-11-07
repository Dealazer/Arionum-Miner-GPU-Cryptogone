//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINERDATA_H
#define ARIONUM_GPU_MINER_MINERDATA_H

#include <string>

#include "block_type.h"

const std::string& blockTypeName(BLOCK_TYPE b);

class MinerData {
private:
    std::string status = "";
    std::string difficulty = "";
    std::string limit = "";
    std::string block = "";
    std::string public_key = "";
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
        std::string s,
        std::string d,
        std::string l,
        std::string b,
        std::string p,
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

    const std::string *getStatus() const;

    const std::string *getDifficulty() const;

    const std::string *getLimit() const;

    const std::string *getBlock() const;

    const std::string *getPublic_key() const;

    uint32_t getHeight() const;

    BLOCK_TYPE getBlockType() const { return type; };
    std::string getArgonPrmsStr() const;

    bool isNewBlock(const std::string *newBlock);

    friend std::ostream &operator<<(std::ostream &os, const MinerData &data);

    friend bool operator==(const MinerData &lhs, const MinerData &rhs);

    friend bool operator!=(const MinerData &lhs, const MinerData &rhs);

    long getLongDiff() const;
};

#endif //ARIONUM_GPU_MINER_MINERDATA_H
