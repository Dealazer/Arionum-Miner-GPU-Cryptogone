//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_STATS_H
#define ARIONUM_GPU_MINER_STATS_H

#include <iostream>
#include <atomic>
#include <chrono>
#include <mutex>
#include <climits>
#include <cmath>

#include "minerdata.h"

using namespace std;

//#define DEBUG_ROUNDS

class Stats {
private:
    std::atomic<long> roundType;
    std::atomic<long> roundHashes;
    std::atomic<double> roundHashRate;
    std::chrono::time_point<std::chrono::system_clock> roundStart;

    std::atomic<long> rounds_cpu;
    std::atomic<long> totalHashes_cpu;
    std::atomic<double> totalTime_cpu_sec;

    std::atomic<long> rounds_gpu;
    std::atomic<long> totalHashes_gpu;
    std::atomic<double> totalTime_gpu_sec;

    std::atomic<long> shares;
    std::atomic<long> blocks;
    std::atomic<long> rejections;

    std::atomic<long> bestDl_cpu;
    std::atomic<long> bestDl_gpu;
    std::atomic<long> blockBestDl;

    std::mutex mutex;

public:

    Stats() :          roundType(-1),
                       roundHashes(0),
                       totalHashes_cpu(0),
                       totalHashes_gpu(0),
                       totalTime_cpu_sec(0.0),
                       totalTime_gpu_sec(0.0),
                       shares(0),
                       blocks(0),
                       rejections(0),
                       rounds_cpu(0),
                       rounds_gpu(0),
                       roundHashRate(0.0),
                       bestDl_cpu(UINT32_MAX),
                       bestDl_gpu(UINT32_MAX),
                       blockBestDl(UINT32_MAX),
                       roundStart(std::chrono::system_clock::now())/*,
                       start(std::chrono::system_clock::now())*/ {
    };

    void addHashes(long hashes);
    bool newShare();
    bool newBlock();
    void newRejection();
    void newDl(long dl, BLOCK_TYPE t);

    void beginRound(const MinerData& data);
    void endRound();

    const atomic<long> &getRounds(BLOCK_TYPE t) const;
    const atomic<long> &getTotalHashes(BLOCK_TYPE t) const;

    const atomic<long> &getRoundHashes() const;    
    const atomic<double> &getRoundHashRate() const;
    const chrono::time_point<chrono::system_clock> &getRoundStart() const;

    double getAvgHashrate(BLOCK_TYPE t) const;
    const atomic<long> &getBestDl(BLOCK_TYPE t) const;
    const atomic<long> &getBlockBestDl() const;

    const atomic<long> &getShares() const;
    const atomic<long> &getBlocks() const;
    const atomic<long> &getRejections() const;

    friend ostream &operator<<(ostream &os, const Stats &stats);

    void blockChange(const MinerData &newData);

private:
    uint32_t rndRange(uint32_t n);
};

#endif //ARIONUM_GPU_MINER_STATS_H
