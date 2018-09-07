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

void Stats::blockChange(const MinerData &newData) {
    s_forceShowHeaders = true;
    if (roundType != -1) {
        endRound();
        blockBestDl = UINT32_MAX;
        beginRound(newData);
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

void Stats::beginRound(const MinerData& data) {
    std::lock_guard<std::mutex> lg(mutex);
    roundType = data.getBlockType();
    roundHashes = 0;    
    roundStart = std::chrono::system_clock::now();
#ifdef DEBUG_ROUNDS
    cout << "---- START ROUND, type=" << roundType << endl;
#endif
}

extern Updater* s_pUpdater;

void Stats::endRound() {
    std::lock_guard<std::mutex> lg(mutex);

    // compute duration
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(now - roundStart);
    auto roundDurationMs = time.count();

    // compute hashrate
    roundHashRate = ((double)roundHashes * 1000.0) / (double)roundDurationMs;

#if (TEST_MODE == TEST_CPU)
    roundType = BLOCK_CPU;
#elif (TEST_MODE == TEST_GPU)
    roundType = BLOCK_GPU;
#endif

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
#ifdef DEBUG_ROUNDS
    cout << "---- END ROUND, duration=" << std::fixed << std::setprecision(1) << roundDurationMs << "ms" << endl;
#endif
}

ostream &operator<<(ostream &os, const Stats &stats) {
    // get current Mining data from updater
    auto data = s_pUpdater->getData();
    
    // -- show header
    const int COL_DATE          = 20;
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

    static unsigned long r = -1;
    r++;
    if (s_forceShowHeaders || (r % 5 == 0)) {

        // s_forceShowHeaders management
        if (s_forceShowHeaders) {
            r = 0;
            s_forceShowHeaders = false;
        }

        // jump a line before header
        cout << endl;

        //
        ostringstream oss_hashrate_instant;
        oss_hashrate_instant << "H/S-" << POOL_UPDATE_RATE_SECONDS << "s";

        // column names
        cout << setw(COL_DATE) << left          << "Date";
#if !TEST_MODE
        cout << setw(COL_HEIGHT) << left        << "Height";
#endif

        cout << setw(COL_TYPE) << left          << "Type";
        cout << setw(COL_HS) << left            << oss_hashrate_instant.str();
        cout << setw(COL_HS_AVG) << left        << "H/S-avg";

#if !TEST_MODE
        cout << setw(COL_SHARES) << left        << "Shares"
             << setw(COL_BLOCKS) << left        << "Blocks"
             << setw(COL_REJECTS) << left       << "Reject"
             << setw(COL_BEST_DL) << left       << "Block best DL"
             << setw(COL_EVER_BEST_DL) << left  << "Ever best DL"
             << setw(COL_MIN_DL) << left        << "Pool min DL";
#endif

        cout << endl;
    }

    // -- show content

    // date
#pragma warning(disable : 4996)
    auto roundStart = stats.getRoundStart();
    auto t = std::chrono::system_clock::to_time_t(roundStart);
    cout << setw(COL_DATE) << left << std::put_time(std::localtime(&t), "%D %T   ");
#pragma warning(default : 4996)

#if !TEST_MODE
    // height
    cout << setw(COL_HEIGHT) << left << (s_pUpdater ? data.getHeight() : (-1));

    // type
    cout << setw(COL_TYPE) << left << (s_pUpdater ? blockTypeName(data.getBlockType()) : "??");

    // mining stats
    bool isMining = stats.getMinerSettings() && stats.getMinerSettings()->mineBlock(data.getBlockType());
    ostringstream oss_hashRate, oss_avgHashRate, ossBlockBestDL, ossEverBestDL;
    if (isMining) {
        oss_hashRate << std::fixed << std::setprecision(1) << stats.getRoundHashRate();
        oss_avgHashRate << std::fixed << std::setprecision(1) << stats.getAvgHashrate(data.getBlockType());
        ossBlockBestDL << stats.getBlockBestDl();
        uint32_t bestEver = stats.getBestDl(data.getBlockType());
        ossEverBestDL << bestEver;
    }
    else {
        oss_hashRate << "off";
        oss_avgHashRate << "off";
        ossBlockBestDL << "N/A";
        ossEverBestDL << "N/A";
    }

    cout << setw(COL_HS)       << left << oss_hashRate.str()
         << setw(COL_HS_AVG)   << left << oss_avgHashRate.str()
         << setw(COL_SHARES)   << left << stats.getShares()
         << setw(COL_BLOCKS)   << left << stats.getBlocks()
         << setw(COL_REJECTS)  << left << stats.getRejections()
         << setw(COL_BEST_DL)  << left << ossBlockBestDL.str()
         << setw(COL_EVER_BEST_DL) << left << ossEverBestDL.str();

    // pool limit
    cout << setw(COL_MIN_DL) << left;
    if (isMining) {
        cout << *data.getLimit();
    }
    else {
        cout << "N/A";
    }
#else
    // test mining stats
    cout << std::fixed << std::setprecision((TEST_MODE == TEST_GPU) ? 1 : 2);
    cout << setw(COL_TYPE)   << left << ((TEST_MODE == TEST_GPU) ? "GPU_T" : "CPU_T")
         << setw(COL_HS)     << left << stats.getRoundHashRate()
         << setw(COL_HS_AVG) << left << stats.getAvgHashrate(((TEST_MODE == TEST_GPU) ? BLOCK_GPU : BLOCK_CPU));
#endif

    cout << endl;
    return os;
}

