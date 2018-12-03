//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_STATS_H
#define ARIONUM_GPU_MINER_STATS_H

#include "minerdata.h"
#include "minersettings.h"
#include <cpprest/http_client.h>
#include <argon2-gpu-common/argon2-common.h>

struct SubmitParams {
    std::string nonce;
    std::string argon;
    std::string public_key;
    bool d;
    bool isBlock;
    BLOCK_TYPE roundType;
    uint32_t dl;
};

class AroMiner;

class Stats {
public:
    Stats(const MinerSettings & minerSettings) :
        minerSettings(minerSettings),
        shares(0),
        blocks(0),
        rejections(0),
        bestDl_cpu(UINT32_MAX),
        bestDl_gpu(UINT32_MAX),
        curBlockBestDl(UINT32_MAX),
        forceShowHeaders(false),
        mutex{} {
    };

    void onMinerTaskStart(AroMiner & miner,
        int minerId, int nMiners, argon2::time_point time);
    void onMinerTaskEnd(
        int minerId, bool hashesAccepted);
    void onMinerDeviceTime(
        int minerId, BLOCK_TYPE t, uint32_t nHashes, std::chrono::nanoseconds duration);
    void onBlockChange(BLOCK_TYPE);
    void onShareFound(const SubmitParams & p);
    void onBlockFound(const SubmitParams & p);
    void onRejectedShare(const SubmitParams & p);
    void onDL(uint32_t dl, BLOCK_TYPE t);

    double lastHashrate() const;
    double averageHashrate(BLOCK_TYPE t) const;
    double maxTheoricalHashrate(BLOCK_TYPE t) const;

    uint32_t bestDL(BLOCK_TYPE t) const;
    uint32_t currentBlockBestDL() const;
    uint64_t sharesFound() const;
    uint64_t blocksFound() const;
    uint64_t sharesRejected() const;

    void printTimePrefix() const;
    void printHeaderTestMode() const;
    void printStatsTestMode() const;
    void printStatsMiningMode(
        const MinerData & data, bool isMining);
    bool dd();

private:
    const MinerSettings & minerSettings;

    uint64_t shares;
    uint64_t blocks;
    uint64_t rejections;

    uint32_t bestDl_cpu;
    uint32_t bestDl_gpu;
    uint32_t curBlockBestDl;

    bool forceShowHeaders;

    std::mutex mutex;

private:
    typedef std::chrono::nanoseconds ns;
    typedef std::chrono::duration<double> fsec;

    struct HashrateAccum {
        uint64_t totalHashes[BLOCK_TYPES_COUNT]{};
        ns totalDuration[BLOCK_TYPES_COUNT]{};

        void addHashes(BLOCK_TYPE bt, uint32_t nHashes, ns duration) {
            totalHashes[bt] += nHashes;
            totalDuration[bt] += duration;
        }

        double average(BLOCK_TYPE bt) const {
            double nSecs = 
                std::chrono::duration_cast<fsec>(totalDuration[bt]).count();
            if (nSecs < 1.0e-6)
                return 0;
            return totalHashes[bt] / nSecs;
        }
    };

    struct MinerStats {
        double lastHashrate() const;
        double averageHashrate(BLOCK_TYPE t) const;

        argon2::time_point lastT{};
        BLOCK_TYPE lastTaskType{ BLOCK_MAX };
        double lastTaskHashrate{ 0.0 };
        bool lastTaskValidated{ false };
        HashrateAccum totalHashrate{};
        HashrateAccum deviceTimeHashrate{};
    };
    
    std::vector<MinerStats> minerStats;

private:
    // stats node
    void nodeSubmitReq(std::string desc, const SubmitParams & p, bool accepted);
    std::stringstream nodeBaseFields(const std::string &query, long roundType);
    std::unique_ptr<web::http::client::http_client> nodeClient();
    void nodeReq(std::string desc, const std::string & paths);
};

#endif //ARIONUM_GPU_MINER_STATS_H
