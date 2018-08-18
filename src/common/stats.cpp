//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include <iostream>
#include "../../include/stats.h"
#include "../../include/updater.h"
#include "../../include/miner.h"
#include <iomanip>

using namespace std;

const atomic<long> &Stats::getRoundHashes() const {
    return roundHashes;
}

const atomic<long> &Stats::getRounds() const {
    return rounds;
}

const atomic<double> &Stats::getHashRate() const {
    return hashRate;
}

const atomic<long> &Stats::getHashes() const {
    return hashes;
}

const atomic<long> &Stats::getShares() const {
    return shares;
}

const atomic<long> &Stats::getBlocks() const {
    return blocks;
}

const atomic<long> &Stats::getBestDl() const {
    return bestDl;
}

const atomic<long> &Stats::getBlockBestDl() const {
    return blockBestDl;
}

const atomic<long> &Stats::getRejections() const {
    return rejections;
}

const chrono::time_point<chrono::system_clock> &Stats::getRoundStart() const {
    return roundStart;
}

const chrono::time_point<chrono::system_clock> &Stats::getStart() const {
    return start;
}

const atomic<double> &Stats::getAvgHashRate() const {
    return avgHashRate;
}

void Stats::addHashes(long newHashes) {
    std::lock_guard<std::mutex> lg(mutex);
    hashes += newHashes;
    roundHashes += newHashes;
}

bool Stats::newShare() {
    std::lock_guard<std::mutex> lg(mutex);
    shares++;
    return (shares % rate) == 0;
}

void Stats::blockChange() {
    blockBestDl = LONG_MAX;
}

bool Stats::newBlock() {
    blocks++;
    return false;
}

void Stats::newRejection() {
    rejections++;
}

void Stats::newDl(long dl) {
    if (dl <= 0)
        return;
    long prev = bestDl;
    if (dl < prev)
        bestDl.compare_exchange_weak(prev, dl);
    long prevBlock = blockBestDl;
    if (dl < prevBlock)
        blockBestDl.compare_exchange_weak(prevBlock, dl);
}

void Stats::newRound() {
    std::lock_guard<std::mutex> lg(mutex);
    // let CUDA / CL warmup the kernels before measuring hashrate
    if (rounds >= 1) { 
        updateHashRate();
    }

    rounds++;
    roundStart = std::chrono::system_clock::now();
    roundHashes = 0;

    // let CUDA / CL warmup the kernels before measuring hashrate
    if (rounds == 1) {
        start = std::chrono::system_clock::now();
        hashes = 0;
    }
}

void Stats::updateHashRate() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(now - roundStart);
    auto roundDuration = time.count();
    if (roundDuration > 0)
        hashRate = (roundHashes * 1000.0) / roundDuration;
    auto globalTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    auto duration = globalTime.count();
    if (duration > 0)
        avgHashRate = (hashes * 1000.0) / duration;
}


#pragma warning(disable:4996)

extern Updater* s_pUpdater;

ostream &operator<<(ostream &os, const Stats &settings) {
    static unsigned long r = -1;
    r++;
    if (r % 10 == 0) {
        cout << endl
             << setw(20) << left << "Date"
#ifdef TEST_GPU_BLOCK
            << setw(12) << left << "Block Type"
            << setw(16) << left << "Hash/s"
#else
             << setw(12) << left << "Height"
             << setw(12) << left << "Block Type"
             << setw(16) << left << "Hash/s"
             << setw(8) << left << "Shares"
             << setw(8) << left << "Blocks"
             << setw(8) << left << "Reject"
             << setw(18) << left << "Block best DL"
             << setw(18) << left << "Ever best DL"
             << setw(18) << left << "Pool min DL"
#endif
             << endl;
    }

    auto roundStart = settings.getRoundStart();
    auto t = std::chrono::system_clock::to_time_t(roundStart);
    
    auto data = s_pUpdater->getData();
    cout << setw(20) << left << std::put_time(std::localtime(&t), "%D %T   ")
#ifdef TEST_GPU_BLOCK
         << setw(12) << left << "GPU TEST"
         << setw(16) << left << settings.getHashRate();
#else
         << setw(12) << left << (s_pUpdater ? data.getHeight() : (-1))
         << setw(12) << left << (s_pUpdater ? blockTypeName(data.getType()) : "??")
         << setw(16) << left << settings.getHashRate()
         << setw(8) << left << settings.getShares()
         << setw(8) << left << settings.getBlocks()
         << setw(8) << left << settings.getRejections()
         << setw(18) << left << settings.getBlockBestDl()
         << setw(18) << left << settings.getBestDl()
         << setw(18) << left << *data.getLimit();
#endif
    return os;
}

