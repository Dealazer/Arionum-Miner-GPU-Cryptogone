//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/stats.h"
#include "../../include/updater.h"
#include "../../include/miner.h"
#include "../../include/testMode.h"
#include "../../include/miner.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <cctype>
#include <chrono>

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

uint32_t rndRange(uint32_t n) {
    static bool s_inited = false;
    static std::mt19937 s_gen;
    static std::uniform_int_distribution<int> s_distrib;
    if (!s_inited) {
        std::unique_ptr<int> dummy(new int[32]);
        unsigned int local = (uintptr_t)(dummy.get()) & 0xFFFFFFFF;

        std::cout << "local = " << local << std::endl;

        unsigned int t = time(0) & 0xFFFFFFFF;
        std::mt19937::result_type seed = (local + t);
        s_gen = std::mt19937(seed);
        s_inited = true;
    }

    s_distrib = std::uniform_int_distribution<int>(0, n - 1);
    return s_distrib(s_gen);
}

double Stats::MinerStats::averageHashrate(BLOCK_TYPE t) const {
    return this->totalHashrate.average(t);
}

double Stats::MinerStats::lastHashrate() const {
    return lastTaskHashrate;
}

void Stats::onMinerTaskStart(AroMiner & miner,
    int minerIndex, int nMiners, argon2::time_point time) {
    std::lock_guard<std::mutex> lg(mutex);

    if (minerStats.size() == 0)
        minerStats.resize(nMiners);
    assert(minerStats.size() == nMiners);

    auto newTaskBlockType = miner.providerBlockType();
    auto &ms = minerStats[minerIndex];
    if (ms.lastT != argon2::time_point()) {
        ns duration = time - ms.lastT;
        if (!ms.lastTaskValidated) {
            ms.lastTaskHashrate = 0;
            ms.totalHashrate.addHashes(ms.lastTaskType, 0, duration);
        }
        else {
            auto nHashes = miner.nHashesPerRun(ms.lastTaskType);
            ms.lastTaskHashrate = (double)(nHashes) / 
                std::chrono::duration_cast<fsec>(duration).count();
            ms.totalHashrate.addHashes(ms.lastTaskType, nHashes, duration);
        }
#if 0
        uint64_t totHashes = 0;
        std::chrono::nanoseconds totDuration{};
        for (const auto& it : minerStats) {
            totHashes += it.totalHashrate.totalHashes[ms.lastTaskType];
            totDuration += it.totalHashrate.totalDuration[ms.lastTaskType];
        }

        std::cout << "miner " << minerIndex << " "
            "TOTAL " << blockTypeName(ms.lastTaskType) <<
            " hashes=" << totHashes <<
            " time=" << std::chrono::duration_cast<fsec>(totDuration).count() <<
            std::endl;
#endif
    }
    ms.lastT = time;
    ms.lastTaskType = newTaskBlockType;
}

void Stats::onMinerTaskEnd(int minerId, bool hashesAccepted) {
    std::lock_guard<std::mutex> lg(mutex);
    minerStats[minerId].lastTaskValidated = hashesAccepted;
}

void Stats::onBlockChange(BLOCK_TYPE blockType) {
    forceShowHeaders = true;
    curBlockBestDl = UINT32_MAX;
}

void Stats::onShareFound(const SubmitParams & p) {
    if (!p.d) {
        std::lock_guard<std::mutex> lg(mutex);
        shares++;
        nodeSubmitReq("share submit (stats node)", p, true);
    }
}

void Stats::onBlockFound(const SubmitParams & p) {
    if (!p.d) {
        std::lock_guard<std::mutex> lg(mutex);
        blocks++;
        nodeSubmitReq("block submit (stats node)", p, true);
    }
}

void Stats::onRejectedShare(const SubmitParams & p) {
    std::lock_guard<std::mutex> lg(mutex);
    if (!p.d) {
        rejections++;
        nodeSubmitReq("reject submit (stats node)", p, false);
    }
}

void Stats::onDL(uint32_t dl, BLOCK_TYPE t) {
    if (dl <= 0)
        return;

    std::lock_guard<std::mutex> lg(mutex);

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
    uint32_t prev = curBlockBestDl;
    if (dl < prev) {
        curBlockBestDl = dl;
    }
}

void Stats::onMinerDeviceTime(
    int minerId, BLOCK_TYPE t, uint32_t nHashes, std::chrono::nanoseconds duration) {
    std::lock_guard<std::mutex> lg(mutex);
    minerStats[minerId].deviceTimeHashrate.addHashes(t, nHashes, duration);

#if 0
    uint64_t totHashes = 0;
    std::chrono::nanoseconds totDuration{};
    for (const auto& it : minerStats) {
        totHashes += it.deviceTimeHashrate.totalHashes[t];
        totDuration += it.deviceTimeHashrate.totalDuration[t];
    }

    std::cout << "miner " << minerId << " "
        "DEVICE " << blockTypeName(t) <<
        " hashes=" << totHashes <<
        " time=" << std::chrono::duration_cast<fsec>(totDuration).count() <<
        std::endl;
#endif
}

double Stats::maxTheoricalHashrate(BLOCK_TYPE bt) const {
    double res = 0;
    for (const auto& it : minerStats) {
        res += it.deviceTimeHashrate.average(bt);
    }
    return res;
}

double Stats::lastHashrate() const {
    double res = 0;
    for (const auto& it : minerStats) {
        res += it.lastHashrate();
    }
    return res;
}

double Stats::averageHashrate(BLOCK_TYPE t) const {
    double res = 0;
    for (const auto& it : minerStats) {
        res += it.averageHashrate(t);
    }
    return res;
}

uint32_t Stats::bestDL(BLOCK_TYPE t) const {
    return (t == BLOCK_GPU) ? bestDl_gpu : bestDl_cpu;
}

uint32_t Stats::currentBlockBestDL() const {
    return curBlockBestDl;
}

uint64_t Stats::sharesFound() const {
    return shares;
}

uint64_t Stats::blocksFound() const {
    return blocks;
}

uint64_t Stats::sharesRejected() const {
    return rejections;
}

bool Stats::dd() {
    auto r = rndRange(DR);
    bool dd = r < DP;
    return dd;
}

void Stats::printTimePrefix() const {
#ifdef WIN32
    #pragma warning(disable : 4996)
#endif
    std::ostringstream oss;
    auto t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    oss << std::setw(COL_TIME) << std::left 
        << std::put_time(std::localtime(&t), "%D %T   ");
    std::cout << oss.str();
#ifdef WIN32
    #pragma warning(default : 4996)
#endif
}

void Stats::printHeaderTestMode() const {
    printTimePrefix();
    std::cout
        << std::setw(10) << std::left << "TYPE"
        << std::setw(10) << std::left << "Instant"
        << std::setw(10) << std::left << "Eff."
        << std::setw(10) << std::left << "Average"
        << std::setw(10) << std::left << "Eff."
        << std::endl;
}

void Stats::printStatsTestMode() const {
    printTimePrefix();

    auto blockType = testModeBlockType();
    double maxTheoricalHs = maxTheoricalHashrate(blockType);
    auto efficiencyStr = [&](double hs) -> std::string {
        if (maxTheoricalHs <= 1e-6)
            return "-";
        double efficiency = std::min((100.0 * hs) / maxTheoricalHs, 100.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << efficiency << "%";
        return oss.str();
    };
    auto avgHs = averageHashrate(blockType);
    auto lastHs = lastHashrate();
    std::cout
        << std::fixed
        << std::setprecision((blockType == BLOCK_GPU) ? 1 : 2)
        << std::setw(10) << std::left << blockTypeName(blockType)
        << std::setw(10) << std::left << lastHs
        << std::setw(10) << std::left << efficiencyStr(lastHs)
        << std::setw(10) << std::left << avgHs
        << std::setw(10) << std::left << efficiencyStr(avgHs)
        << std::endl;
}

void Stats::printStatsMiningMode(
    const MinerData & data, bool isMining) {
    static unsigned long r = -1;
    r++;
    if (forceShowHeaders || (r % 5 == 0)) {
        if (forceShowHeaders) {
            r = 0;
            forceShowHeaders = false;
        }
        std::cout
            << std::endl
            << std::setw(COL_TIME) << std::left << "Date"
            << std::setw(COL_HEIGHT) << std::left << "Height"
            << std::setw(COL_TYPE) << std::left << "Type"
            << std::setw(COL_HS) << std::left << "H/S-last"
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
        oss_hashRate << std::fixed << std::setprecision(1) 
            << lastHashrate();
        oss_avgHashRate << std::fixed << std::setprecision(1)
            << averageHashrate(blockType);
        ossBlockBestDL << currentBlockBestDL();
        ossEverBestDL << bestDL(blockType);
    }
    else {
        oss_hashRate << "off";
        oss_avgHashRate << "off";
        ossBlockBestDL << "N/A";
        ossEverBestDL << "N/A";
    }

    std::cout << std::setw(COL_HS) << std::left << oss_hashRate.str()
        << std::setw(COL_HS_AVG) << std::left << oss_avgHashRate.str()
        << std::setw(COL_SHARES) << std::left << sharesFound()
        << std::setw(COL_BLOCKS) << std::left << blocksFound()
        << std::setw(COL_REJECTS) << std::left << sharesRejected()
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
        [](char const& c) -> bool {
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
        auto showStatsError = [](const std::string & desc, const std::string & type, const std::string & err) -> void {
            static std::chrono::time_point<std::chrono::system_clock> s_lastT;
            auto durationSinceLastPrint = std::chrono::system_clock::now() - s_lastT;
            const auto STATS_ERROR_PRINT_INTERVAL_SECONDS = std::chrono::seconds(30);
            if (durationSinceLastPrint >= STATS_ERROR_PRINT_INTERVAL_SECONDS) {
                std::cout << "-- " << desc << " " <<  type << " => " << err << std::endl;
                s_lastT = std::chrono::system_clock::now();
            }
        };

        try
        {
            auto status = response.get().status_code();
            if (status != status_codes::OK)
                showStatsError(desc, "status code != OK", std::to_string(status));
        }
        catch (const web::http::http_exception & e) {
            showStatsError(desc, "http exception", e.what());
        }
        catch (const std::exception e) {
            showStatsError(desc, "exception", e.what());
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