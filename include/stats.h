//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_STATS_H
#define ARIONUM_GPU_MINER_STATS_H

#include "minerdata.h"
#include "minersettings.h"
#include <cpprest/http_client.h>

struct SubmitParams {
    std::string nonce;
    std::string argon;
    std::string public_key;
    bool d;
    bool isBlock;
    BLOCK_TYPE roundType;
    uint32_t dl;
};

class Stats {
private:
    const MinerSettings & minerSettings;

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

    Stats(const MinerSettings & minerSettings) :
        minerSettings(minerSettings),
        roundType(-1),
        roundHashes(0),
        roundHashRate(0.0),
        roundStart(std::chrono::system_clock::now()),
        rounds_cpu(0),
        totalHashes_cpu(0),
        totalTime_cpu_sec(0.0),
        rounds_gpu(0),
        totalHashes_gpu(0),
        totalTime_gpu_sec(0.0),
        shares(0),
        blocks(0),
        rejections(0),
        bestDl_cpu(UINT32_MAX),
        bestDl_gpu(UINT32_MAX),
        blockBestDl(UINT32_MAX),
        mutex{} {
    };

    bool dd();
    void addHashes(long hashes);
    void newShare(const SubmitParams & p);
    void newBlock(const SubmitParams & p);
    void newRejection(const SubmitParams & p);
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

    const std::atomic<long> &getRounds(BLOCK_TYPE t) const;
    const std::atomic<long> &getTotalHashes(BLOCK_TYPE t) const;

    const std::atomic<long> &getRoundHashes() const;
    const std::atomic<double> &getRoundHashRate() const;
    const std::chrono::time_point<std::chrono::system_clock> &getRoundStart() const;

    double getAvgHashrate(BLOCK_TYPE t) const;
    const std::atomic<uint32_t> &getBestDl(BLOCK_TYPE t) const;
    const std::atomic<uint32_t> &getBlockBestDl() const;

    const std::atomic<long> &getShares() const;
    const std::atomic<long> &getBlocks() const;
    const std::atomic<long> &getRejections() const;

    void blockChange(BLOCK_TYPE blockType);

private:
    void nodeSubmitReq(std::string desc, const SubmitParams & p, bool accepted);
    std::stringstream nodeBaseFields(const std::string &query, long roundType);
    std::unique_ptr<web::http::client::http_client> nodeClient();
    void nodeReq(std::string desc, const std::string & paths);

    uint32_t rndRange(uint32_t n);
};

#endif //ARIONUM_GPU_MINER_STATS_H
