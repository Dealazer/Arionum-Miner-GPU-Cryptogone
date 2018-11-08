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

    std::atomic<uint32_t> bestDl_cpu;
    std::atomic<uint32_t> bestDl_gpu;
    std::atomic<uint32_t> blockBestDl;

    std::mutex mutex;

public:

    Stats() :
        roundType(-1),
        roundHashes(0),
        roundStart(std::chrono::system_clock::now()),
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
        blockBestDl(UINT32_MAX) {
    };

    bool dd();
    void addHashes(long hashes);
    void newShare(bool dd);
    void newBlock(bool dd);
    void newRejection();
    void newDl(uint32_t dl, BLOCK_TYPE t);

    void beginRound(BLOCK_TYPE blockType);
    void endRound();

    void printTimePrefix() const;
    void printRoundStats(float nSeconds) const;
    void printMiningStats(
        const MinerData & data,
        bool useLastHashrateInsteadOfRoundAvg,
        bool isMining);

    void printRoundStatsHeader() const;

    const atomic<long> &getRounds(BLOCK_TYPE t) const;
    const atomic<long> &getTotalHashes(BLOCK_TYPE t) const;

    const atomic<long> &getRoundHashes() const;
    const atomic<double> &getRoundHashRate() const;
    const chrono::time_point<chrono::system_clock> &getRoundStart() const;

    double getAvgHashrate(BLOCK_TYPE t) const;
    const atomic<uint32_t> &getBestDl(BLOCK_TYPE t) const;
    const atomic<uint32_t> &getBlockBestDl() const;

    const atomic<long> &getShares() const;
    const atomic<long> &getBlocks() const;
    const atomic<long> &getRejections() const;

    void blockChange(BLOCK_TYPE blockType);

private:
    uint32_t rndRange(uint32_t n);
};

#endif //ARIONUM_GPU_MINER_STATS_H
