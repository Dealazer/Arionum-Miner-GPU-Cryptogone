//
// Created by guli on 31/01/18.
//
#include <iostream>
#include "../../include/stats.h"
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

const atomic<long> &Stats::getRejections() const {
    return rejections;
}

const chrono::time_point<chrono::high_resolution_clock> &Stats::getRoundStart() const {
    return roundStart;
}

const chrono::time_point<chrono::high_resolution_clock> &Stats::getStart() const {
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

void Stats::newShare() {
    shares++;
}

void Stats::newBlock() {
    blocks++;
}

void Stats::newRejection() {
    rejections++;
}

void Stats::newDl(long dl) {
    long prev = bestDl;
    cout << dl << endl;
    cout << prev << endl;
    if (dl > 0 && dl < prev)
        bestDl.compare_exchange_weak(prev, dl);
}

void Stats::newRound() {
    std::lock_guard<std::mutex> lg(mutex);
    updateHashRate();
    rounds++;
    roundStart = std::chrono::high_resolution_clock::now();
    roundHashes = 0;
}

void Stats::updateHashRate() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(now - roundStart);
    long roundDuration = time.count();
    if (roundDuration > 0)
        hashRate = (roundHashes * 1000.0) / roundDuration;
    auto globalTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    long duration = globalTime.count();
    if (duration > 0)
        avgHashRate = (hashes * 1000.0) / duration;
}

ostream &operator<<(ostream &os, const Stats &settings) {
    if (settings.getRounds() % 10 == 0) {
        cout << endl
             << setw(20) << left << "Date"
             << setw(16) << left << "Avg hash/s"
             << setw(16) << left << "Hash/s"
             << setw(20) << left << "Total hashes"
             << setw(8) << left << "Shares"
             << setw(8) << left << "Blocks"
             << setw(8) << left << "Reject"
             << setw(14) << left << "Best DL"
             << endl;
    }
    auto t = std::chrono::system_clock::to_time_t(settings.getRoundStart());
    cout << setw(20) << left << std::put_time(std::localtime(&t), "%D %T   ")
         << setw(16) << left << settings.getAvgHashRate()
         << setw(16) << left << settings.getHashRate()
         << setw(20) << left << settings.getHashes()
         << setw(8) << left << settings.getShares()
         << setw(8) << left << settings.getBlocks()
         << setw(8) << left << settings.getRejections()
         << setw(14) << left << settings.getBestDl();
    return os;
}

