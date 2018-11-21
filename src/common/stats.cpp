//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/stats.h"
#include "../../include/updater.h"
#include "../../include/miner.h"
#include "../../include/testMode.h"
#include "../../include/miners_stats.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <cctype>

const bool DEBUG_ROUNDS = false;
const int COL_TIME = 20;
const int COL_TYPE = 8;
const int COL_HEIGHT = 8;
const int COL_HS = 10;
const int COL_HS_AVG = 10;
const int COL_SHARES = 8;
const int COL_BLOCKS = 8;
const int COL_REJECTS = 8;
const int COL_BEST_DL = 16;
const int COL_EVER_BEST_DL = 16;
const int COL_MIN_DL = 16;
const uint32_t DP = 100;
const uint32_t DR = 10000;

static bool s_forceShowHeaders = false;

const std::atomic<long> &Stats::getRoundHashes() const {
    return roundHashes;
}

const std::atomic<long> &Stats::getRounds(BLOCK_TYPE t) const {
    return (t == BLOCK_GPU) ? rounds_gpu : rounds_cpu;
}

const std::atomic<double> &Stats::getRoundHashRate() const {
    return roundHashRate;
}

const std::atomic<long> &Stats::getTotalHashes(BLOCK_TYPE t) const {
    return (t == BLOCK_GPU) ? totalHashes_gpu : totalHashes_cpu;
}

const std::atomic<uint32_t> &Stats::getBestDl(BLOCK_TYPE t) const {
    return (t == BLOCK_GPU) ? bestDl_gpu : bestDl_cpu;
}

double Stats::getAvgHashrate(BLOCK_TYPE t) const {
    if (t == BLOCK_CPU) {
        return ((totalHashes_cpu > 0) ? 
            ((double)totalHashes_cpu / totalTime_cpu_sec) : 0.0);
    }
    else if (t == BLOCK_GPU) {
        return ((totalHashes_gpu > 0) ? 
            ((double)totalHashes_gpu / totalTime_gpu_sec) : 0.0);
    }
    return 0.0;
}

const std::atomic<long> &Stats::getShares() const {
    return shares;
}

const std::atomic<long> &Stats::getBlocks() const {
    return blocks;
}

const std::atomic<uint32_t> &Stats::getBlockBestDl() const {
    return blockBestDl;
}

const std::atomic<long> &Stats::getRejections() const {
    return rejections;
}

const std::chrono::time_point<std::chrono::system_clock> &Stats::getRoundStart() const {
    return roundStart;
}

uint32_t Stats::rndRange(uint32_t n) {
    static bool s_inited = false;
    static std::mt19937 s_gen;
    static std::uniform_int_distribution<int> s_distrib;
    if (!s_inited) {
        unsigned int local = (uintptr_t)(this) & 0xFFFFFFFF;
        unsigned int t = time(0) & 0xFFFFFFFF;
        std::mt19937::result_type seed = (local + t);
        s_gen = std::mt19937(seed);
        s_inited = true;
    }

    s_distrib = std::uniform_int_distribution<int>(0, n - 1);
    return s_distrib(s_gen);
}

void Stats::addHashes(long newHashes) {
    std::lock_guard<std::mutex> lg(mutex);
    roundHashes += newHashes;
}

void Stats::newShare(const SubmitParams & p) {
    if (!p.d) {
        std::lock_guard<std::mutex> lg(mutex);
        shares++;
        nodeSubmitReq("share submit (stats node)", p, true);
    }
}

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

void Stats::newBlock(const SubmitParams & p) {
    if (!p.d) {
        std::lock_guard<std::mutex> lg(mutex);
        blocks++;
        nodeSubmitReq("block submit (stats node)", p, true);
    }
}

void Stats::newRejection(const SubmitParams & p) {
    rejections++;
    nodeSubmitReq("reject submit (stats node)", p, false);
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
    if (DEBUG_ROUNDS) {
        printTimePrefix();
        std::cout << "---- START ROUND, type=" << roundType << std::endl;
    }
}

void Stats::endRound() {
    std::lock_guard<std::mutex> lg(mutex);

    // compute duration
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(now - roundStart);
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
    if (DEBUG_ROUNDS)
        std::cout << "---- END ROUND, duration=" << std::fixed << std::setprecision(1)
            << roundDurationMs << "ms" << std::endl;

    // send stats to node
    if (!testMode() && minerSettings.hasStatsNode()) {
        std::stringstream paths = nodeBaseFields("report", roundType);
        paths << "&hashes=" << roundHashes << "&elapsed=" << roundDurationMs;
        nodeReq("stats report", paths.str());
    }
}

void Stats::printTimePrefix() const {
#ifdef WIN32
    #pragma warning(disable : 4996)
#endif
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::cout << std::setw(COL_TIME) << std::left << std::put_time(std::localtime(&t), "%D %T   ");
#ifdef WIN32
    #pragma warning(default : 4996)
#endif
}

void Stats::printRoundStatsHeader() const {
    printTimePrefix();
    std::cout
        << std::setw(10) << std::left << "TYPE"
        << std::setw(10) << std::left << "Instant"
        << std::setw(10) << std::left << "Average"
        << std::setw(10) << std::left << "Hashes"
        << std::endl;
}

void Stats::printRoundStats(float nSeconds) const {
    printTimePrefix();

    auto blockType = testModeBlockType();
    std::ostringstream hashes;
    hashes << getRoundHashes() << " in " << std::fixed << std::setprecision(2) << nSeconds << "s";
    double hashRateInstant = double(getRoundHashes()) / nSeconds;
    std::cout
        << std::fixed
        << std::setprecision((blockType == BLOCK_GPU) ? 1 : 2)
        << std::setw(10) << std::left << blockTypeName(blockType)
        << std::setw(10) << std::left << hashRateInstant
        << std::setw(10) << std::left << getAvgHashrate(blockType)
        << std::setw(10) << std::left << hashes.str()
        << std::endl;
}

void Stats::printMiningStats(const MinerData & data, 
    bool useLastHashrateInsteadOfRoundAvg, bool isMining) {
    static unsigned long r = -1;
    r++;
    if (s_forceShowHeaders || (r % 5 == 0)) {
        if (s_forceShowHeaders) {
            r = 0;
            s_forceShowHeaders = false;
        }

        std::ostringstream oss_hashrate_instant;
        if (useLastHashrateInsteadOfRoundAvg) {
            oss_hashrate_instant 
                << "H/S-last";
        }
        else {
            oss_hashrate_instant
                << "H/S-" << POOL_UPDATE_RATE_SECONDS << "s";
        }

        std::cout
            << std::endl
            << std::setw(COL_TIME) << std::left << "Date"
            << std::setw(COL_HEIGHT) << std::left << "Height"
            << std::setw(COL_TYPE) << std::left << "Type"
            << std::setw(COL_HS) << std::left << oss_hashrate_instant.str()
            << std::setw(COL_HS_AVG) << std::left << "H/S-avg"
            << std::setw(COL_SHARES) << std::left << "Shares"
            << std::setw(COL_BLOCKS) << std::left << "Blocks"
            << std::setw(COL_REJECTS) << std::left << "Reject"
            << std::setw(COL_BEST_DL) << std::left << "Block best DL"
            << std::setw(COL_EVER_BEST_DL) << std::left << "Ever best DL"
            << std::setw(COL_MIN_DL) << std::left << "Pool min DL"
            << std::endl;
    }

    printTimePrefix();

    auto blockType = data.getBlockType();
    std::cout << std::setw(COL_HEIGHT) << std::left << data.getHeight();
    std::cout << std::setw(COL_TYPE) << std::left << blockTypeName(blockType);
 
    std::ostringstream
        oss_hashRate, oss_avgHashRate, 
        ossBlockBestDL, ossEverBestDL;
    if (isMining) {
        oss_hashRate 
            << std::fixed << std::setprecision(1) 
            << (useLastHashrateInsteadOfRoundAvg ?
                    minerStatsGetLastHashrate(blockType) : getRoundHashRate().load());
        oss_avgHashRate << std::fixed << std::setprecision(1)
            << getAvgHashrate(blockType);
        ossBlockBestDL 
            << getBlockBestDl();
        std::uint32_t bestEver =
            getBestDl(blockType);
        ossEverBestDL << bestEver;
    }
    else {
        oss_hashRate << "off";
        oss_avgHashRate << "off";
        ossBlockBestDL << "N/A";
        ossEverBestDL << "N/A";
    }

    std::cout << std::setw(COL_HS) << std::left << oss_hashRate.str()
        << std::setw(COL_HS_AVG) << std::left << oss_avgHashRate.str()
        << std::setw(COL_SHARES) << std::left << getShares()
        << std::setw(COL_BLOCKS) << std::left << getBlocks()
        << std::setw(COL_REJECTS) << std::left << getRejections()
        << std::setw(COL_BEST_DL) << std::left << ossBlockBestDL.str()
        << std::setw(COL_EVER_BEST_DL) << std::left << ossEverBestDL.str();

    std::cout << std::setw(COL_MIN_DL) << std::left;
    std::cout << (isMining ? *data.getLimit() : "N/A");

    std::cout << std::endl;
}

std::stringstream Stats::nodeBaseFields(const std::string &query, long roundType) {
    std::stringstream paths;
    std::string blockTypeStr = (roundType == BLOCK_CPU) ? "CPU" : "GPU";

    std::string encodedWorkerId = minerSettings.uniqueID();
    encodedWorkerId.erase(std::remove_if(encodedWorkerId.begin(), encodedWorkerId.end(), 
        [](auto const& c) -> bool {
        bool keep = std::isalnum(c) ||
            c == '-' || c == '.' || c == '_' || c == '~';
        return !keep;
    }), encodedWorkerId.end());
    
    paths << "/report.php?q=" << query
        << "&token=" << minerSettings.statsToken()
        << "&id=" << encodedWorkerId << blockTypeStr
        << "&type=" << "arionumGPUminer" << blockTypeStr;
    return paths;
}

std::unique_ptr<http_client> Stats::nodeClient() {
    http_client_config config;
    config.set_timeout(utility::seconds(2));
    auto statsAdress = toUtilityString(minerSettings.statsAPIUrl());
    std::unique_ptr<http_client> p(new http_client(statsAdress, config));
    return p;
}

void Stats::nodeReq(std::string desc, const std::string & paths) {
    //std::cout << paths << std::endl;

    auto client = nodeClient();
    http_request req(methods::GET);
    auto _paths = toUtilityString(paths);
    req.set_request_uri(_paths.data());
    client->request(req)
        .then([desc](pplx::task<web::http::http_response> response)
    {
        try
        {
            if (response.get().status_code() != status_codes::OK)
                std::cout << "-- " << desc << " error" << std::endl;
        }
        catch (const std::exception e) {
            std::cout << "-- " << desc << " exception: " << e.what() << std::endl;
        }
    });
}

void Stats::nodeSubmitReq(std::string desc, const SubmitParams & p, bool accepted) {
    if (!minerSettings.hasStatsNode())
        return;

    // currently missing:
    // p.nonce,  p.argon: need UTF8 encode, difficulty
    std::stringstream paths = nodeBaseFields("discovery", p.roundType);
    paths << "&retries=" << 0 << "&dl=" << p.dl;
    if (accepted)
        paths << "&confirmed";

    nodeReq(desc, paths.str());
}