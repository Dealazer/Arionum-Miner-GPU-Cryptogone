//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include <iostream>
#include "../../include/stats.h"
#include "../../include/updater.h"
#include "../../include/miner.h"
#include <iomanip>
#include <sstream>
#include <random>

using namespace std;
using std::cout;

#define DEBUG_ROUNDS (0)

const atomic<long> &Stats::getRoundHashes() const {
    return roundHashes;
}

const atomic<long> &Stats::getRounds(BLOCK_TYPE t) const {
    return (t == BLOCK_GPU) ? rounds_gpu : rounds_cpu;
}

const atomic<double> &Stats::getRoundHashRate() const {
    return roundHashRate;
}

const atomic<long> &Stats::getTotalHashes(BLOCK_TYPE t) const {
    return (t == BLOCK_GPU) ? totalHashes_gpu : totalHashes_cpu;
}

const atomic<uint32_t> &Stats::getBestDl(BLOCK_TYPE t) const {
    return (t == BLOCK_GPU) ? bestDl_gpu : bestDl_cpu;
}

double Stats::getAvgHashrate(BLOCK_TYPE t) const
{
    if (t == BLOCK_CPU) {
        return ((totalHashes_cpu > 0) ? ((double)totalHashes_cpu / totalTime_cpu_sec) : 0.0);
    }
    else if (t == BLOCK_GPU) {
        return ((totalHashes_gpu > 0) ? ((double)totalHashes_gpu / totalTime_gpu_sec) : 0.0);
    }
    return 0.0;
}

const atomic<long> &Stats::getShares() const {
    return shares;
}

const atomic<long> &Stats::getBlocks() const {
    return blocks;
}

const atomic<uint32_t> &Stats::getBlockBestDl() const {
    return blockBestDl;
}

const atomic<long> &Stats::getRejections() const {
    return rejections;
}

const chrono::time_point<chrono::system_clock> &Stats::getRoundStart() const {
    return roundStart;
}

uint32_t Stats::rndRange(uint32_t n) {
    static bool s_inited = false;
    static mt19937 s_gen;
    static std::uniform_int_distribution<int> s_distrib;
    if (!s_inited) {
        unsigned int local = (uintptr_t)(this) & 0xFFFFFFFF;
        unsigned int t = time(0) & 0xFFFFFFFF;
        mt19937::result_type seed = (local + t);
        s_gen = mt19937(seed);
        s_inited = true;
    }

    s_distrib = std::uniform_int_distribution<int>(0, n - 1);
    return s_distrib(s_gen);
}

void Stats::addHashes(long newHashes) {
    std::lock_guard<std::mutex> lg(mutex);
    roundHashes += newHashes;
}

const uint32_t DP = 100;
const uint32_t DR = 10000;

void Stats::newShare(bool dd) {
    if (!dd) {
        std::lock_guard<std::mutex> lg(mutex);
        shares++;
    }
}

static bool s_forceShowHeaders = false;

bool Stats::dd() {
    auto r = rndRange(DR);
    bool dd = r < DP;
    return dd;
}

void Stats::blockChange(BLOCK_TYPE blockType) {
    s_forceShowHeaders = true;
    if (roundType != -1) {
        endRound();
        blockBestDl = UINT32_MAX;
        beginRound(blockType);
    }
}

void Stats::newBlock(bool dd) {
    if (!dd) {
        std::lock_guard<std::mutex> lg(mutex);
        blocks++;
    }
}

void Stats::newRejection() {
    rejections++;
}

void Stats::newDl(uint32_t dl, BLOCK_TYPE t) {
    if (dl <= 0)
        return;

    // update best ever dl
    if (t == BLOCK_CPU) {
        uint32_t prev = bestDl_cpu;
        if (dl < prev)
            bestDl_cpu = dl;
    }
    else if (t == BLOCK_GPU) {
        uint32_t prev = bestDl_gpu;
        if (dl < prev)
            bestDl_gpu = dl;
    }

    // update cur block best dl
    uint32_t prev = blockBestDl;
    if (dl < prev) {
        blockBestDl = dl;
    }
}

void Stats::beginRound(BLOCK_TYPE blockType) {
    std::lock_guard<std::mutex> lg(mutex);
    roundType = blockType;
    roundHashes = 0;    
    roundStart = std::chrono::system_clock::now();
#if DEBUG_ROUNDS
    cout << "---- START ROUND, type=" << roundType << endl;
#endif
}

extern Updater* s_pUpdater;

void Stats::endRound() {
    std::lock_guard<std::mutex> lg(mutex);

    // compute duration
    auto now = chrono::system_clock::now();
    auto time = chrono::duration_cast<std::chrono::milliseconds>(now - roundStart);
    auto roundDurationMs = time.count();

    // compute hashrate
    roundHashRate = ((double)roundHashes * 1000.0) / (double)roundDurationMs;

    if (testMode()) {
        roundType = testModeBlockType();
    }

    // record stats for averages
    if (roundType == BLOCK_GPU) {
        rounds_gpu++;

        totalHashes_gpu += roundHashes;
        totalTime_gpu_sec = totalTime_gpu_sec + (double)roundDurationMs / 1000.0;
    }
    else {
        rounds_cpu++;
        
        totalHashes_cpu += roundHashes;
        totalTime_cpu_sec = totalTime_cpu_sec + (double)roundDurationMs / 1000.0;
    }
#if DEBUG_ROUNDS
    cout << "---- END ROUND, duration=" << std::fixed << std::setprecision(1) 
        << roundDurationMs << "ms" << endl;
#endif
}

const int COL_TIME = 20;

void Stats::printTimePrefix() const {
#pragma warning(disable : 4996)
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    cout << setw(COL_TIME) << left << std::put_time(std::localtime(&t), "%D %T   ");
#pragma warning(default : 4996)
}

void Stats::printRoundStatsHeader() const {
    printTimePrefix();
    cout
        << setw(10) << left << "TYPE"
        << setw(10) << left << "Instant"
        << setw(10) << left << "Average"
        << setw(10) << left << "Hashes"
        << endl;
}

void Stats::printRoundStats(float nSeconds) const {
    printTimePrefix();

    auto blockType = testModeBlockType();
    ostringstream hashes;
    hashes << getRoundHashes() << " in " << std::fixed << std::setprecision(2) << nSeconds << "s";
    double hashRateInstant = double(getRoundHashes()) / nSeconds;
    cout
        << std::fixed
        << std::setprecision((blockType == BLOCK_GPU) ? 1 : 2)
        << setw(10) << left << blockTypeName(blockType)
        << setw(10) << left << hashRateInstant
        << setw(10) << left << getAvgHashrate(blockType)
        << setw(10) << left << hashes.str()
        << endl;
}

ostream &operator<<(ostream &os, const Stats &stats) {
    const int COL_TYPE          = 8;
    const int COL_HEIGHT        = 8;
    const int COL_HS            = 10;
    const int COL_HS_AVG        = 10;
    const int COL_SHARES        = 8;
    const int COL_BLOCKS        = 8;
    const int COL_REJECTS       = 8;
    const int COL_BEST_DL       = 16;
    const int COL_EVER_BEST_DL  = 16;
    const int COL_MIN_DL        = 16;

    auto data = s_pUpdater->getData();

    bool useNewHashrateDisplay = 
        stats.getMinerSettings()->useLastHashrateInsteadOfRoundAvg();

    static unsigned long r = -1;
    r++;
    if (s_forceShowHeaders || (r % 5 == 0)) {
        if (s_forceShowHeaders) {
            r = 0;
            s_forceShowHeaders = false;
        }

        ostringstream oss_hashrate_instant;
        if (useNewHashrateDisplay) {
            oss_hashrate_instant 
                << "H/S-last";
        }
        else {
            oss_hashrate_instant
                << "H/S-" << POOL_UPDATE_RATE_SECONDS << "s";
        }

        cout 
            << endl
            << setw(COL_TIME) << left << "Date"
            << setw(COL_HEIGHT) << left << "Height"
            << setw(COL_TYPE) << left << "Type"
            << setw(COL_HS) << left << oss_hashrate_instant.str()
            << setw(COL_HS_AVG) << left << "H/S-avg"
            << setw(COL_SHARES) << left << "Shares"
            << setw(COL_BLOCKS) << left << "Blocks"
            << setw(COL_REJECTS) << left << "Reject"
            << setw(COL_BEST_DL) << left << "Block best DL"
            << setw(COL_EVER_BEST_DL) << left << "Ever best DL"
            << setw(COL_MIN_DL) << left << "Pool min DL"
            << endl;
    }

    stats.printTimePrefix();

    auto blockType = 
        data.getBlockType();

    cout << setw(COL_HEIGHT) << left << 
        (s_pUpdater ? data.getHeight() : (-1));
    cout << setw(COL_TYPE) << left << 
        (s_pUpdater ? blockTypeName(blockType) : "??");
 
    ostringstream 
        oss_hashRate, oss_avgHashRate, 
        ossBlockBestDL, ossEverBestDL;
    bool isMining = 
        stats.getMinerSettings() && 
        stats.getMinerSettings()->mineBlock(blockType);
    if (isMining) {
        oss_hashRate 
            << std::fixed << std::setprecision(1) 
            << (useNewHashrateDisplay ? 
                    minerStatsGetLastHashrate() : stats.getRoundHashRate().load());
        oss_avgHashRate << std::fixed << std::setprecision(1)
            << stats.getAvgHashrate(blockType);
        ossBlockBestDL 
            << stats.getBlockBestDl();
        uint32_t bestEver = 
            stats.getBestDl(blockType);
        ossEverBestDL << bestEver;
    }
    else {
        oss_hashRate << "off";
        oss_avgHashRate << "off";
        ossBlockBestDL << "N/A";
        ossEverBestDL << "N/A";
    }

    cout << setw(COL_HS) << left << oss_hashRate.str()
        << setw(COL_HS_AVG) << left << oss_avgHashRate.str()
        << setw(COL_SHARES) << left << stats.getShares()
        << setw(COL_BLOCKS) << left << stats.getBlocks()
        << setw(COL_REJECTS) << left << stats.getRejections()
        << setw(COL_BEST_DL) << left << ossBlockBestDL.str()
        << setw(COL_EVER_BEST_DL) << left << ossEverBestDL.str();

    cout << setw(COL_MIN_DL) << left;
    cout << (isMining ? *data.getLimit() : "N/A");

    cout << endl;
    return os;
}

