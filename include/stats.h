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

using namespace std;

class Stats {
private:
    std::atomic<long> roundHashes;
    std::atomic<long> rounds;
    std::atomic<double> hashRate;
    std::atomic<double> avgHashRate;
    std::atomic<long> hashes;
    std::atomic<long> shares;
    std::atomic<long> blocks;
    std::atomic<long> rejections;
    std::atomic<long> bestDl;
    std::atomic<long> blockBestDl;
    std::chrono::time_point<std::chrono::system_clock> roundStart;
    std::chrono::time_point<std::chrono::system_clock> start;
    int rate;
    std::mutex mutex;

    void updateHashRate();

public:

    Stats(double dd) : roundHashes(0),
                       hashes(0),
                       shares(0),
                       blocks(0),
                       rejections(0),
                       rounds(-1),
                       hashRate(0.0),
                       avgHashRate(0.0),
                       bestDl(LONG_MAX),
                       blockBestDl(LONG_MAX),
                       roundStart(std::chrono::system_clock::now()),
                       start(std::chrono::system_clock::now()) {
        double t = dd <= 0.5 ? 0.5 : dd;
        rate = (int)std::round(100 / t);
    };

    void addHashes(long hashes);

    bool newShare();

    bool newBlock();

    void newRejection();

    void newDl(long dl);

    void newRound();

    const atomic<long> &getRoundHashes() const;

    const atomic<long> &getRounds() const;

    const atomic<double> &getHashRate() const;

    const atomic<long> &getHashes() const;

    const atomic<long> &getShares() const;

    const atomic<long> &getBestDl() const;

    const atomic<long> &getBlockBestDl() const;

    const atomic<long> &getBlocks() const;

    const atomic<long> &getRejections() const;

    const chrono::time_point<chrono::system_clock> &getRoundStart() const;

    const chrono::time_point<chrono::system_clock> &getStart() const;

    friend ostream &operator<<(ostream &os, const Stats &stats);

    const atomic<double> &getAvgHashRate() const;

    void blockChange();
};

#endif //ARIONUM_GPU_MINER_STATS_H
