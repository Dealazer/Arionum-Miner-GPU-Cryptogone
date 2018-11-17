//
// Created by guli on 01/02/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/miner.h"
#include "../../include/aro_tools.h"
#include "../../include/perfscope.h"
#include "../../include/to_base64.h"
#include <argon2.h>
#include "../../argon2/src/core.h"

#include <openssl/sha.h>
#include <sstream>
#include <iomanip>
#include <thread>
#include <map>

const auto INITIAL_BLOCK_TYPE = BLOCK_GPU;

AroMiner::AroMiner(
    const argon2::MemConfig &cfg, const Services& services, 
    argon2::OPT_MODE cpuOptimizationMode) :
    services(services), 
    memConfig(cfg), 
    argon_params{},
    optPrms(configureArgon(
        AroConfig::passes(INITIAL_BLOCK_TYPE),
        AroConfig::memCost(INITIAL_BLOCK_TYPE),
        AroConfig::lanes(INITIAL_BLOCK_TYPE))),
    miningConfig(memConfig, *argon_params, optPrms, INITIAL_BLOCK_TYPE),
    nonces{},
    cpuBlocksOptimizationMode(cpuOptimizationMode) {
}

uint32_t AroMiner::nHashesPerRun() const {
    uint32_t nHashes = 0;
    auto blockType = services.nonceProvider.currentBlockType();
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++)
        nHashes += (uint32_t)memConfig.batchSizes[blockType][i];
    return nHashes;
}

argon2::OPT_MODE AroMiner::optimizationMode(BLOCK_TYPE blockType) const {
    return (blockType == BLOCK_GPU) ? argon2::BASELINE : cpuBlocksOptimizationMode;
}

bool AroMiner::needReconfigure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes) const {
    return
        argon_params->getTimeCost() != t_cost ||
        argon_params->getMemoryCost() != m_cost ||
        argon_params->getLanes() != lanes;
}

argon2::OptParams AroMiner::configureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes) {
    PERFSCOPE("Miner::configure");
    BLOCK_TYPE bt = (lanes == 1 && t_cost == 1) ? BLOCK_CPU : BLOCK_GPU;
    auto & salt = services.nonceProvider.salt(bt);
    argon_params.reset(new argon2::Argon2Params(
        32, salt.data(), 16, nullptr, 0, nullptr, 0, t_cost, m_cost, lanes));

    argon2::OptParams optPrms;
    if (bt == BLOCK_CPU)
        optPrms = precomputeArgon(argon_params.get());
    optPrms.mode = optimizationMode(bt);
    return optPrms;
}

bool AroMiner::updateNonceProvider() {
    bool ok = services.nonceProvider.update();
    auto bt = services.nonceProvider.currentBlockType();
    if (ok) {
        auto t_cost = AroConfig::passes(bt);
        auto m_cost = AroConfig::memCost(bt);
        auto lanes = AroConfig::lanes(bt);
        if (!needReconfigure(t_cost, m_cost, lanes))
            return ok;
        optPrms = configureArgon(t_cost, m_cost, lanes);
        miningConfig = argon2::Argon2iMiningConfig(
            memConfig, *argon_params, optPrms, (int)bt);
        reconfigureKernel();
    }
    return ok;
}

bool AroMiner::generateNonces() {
    auto nHashes = nHashesPerRun();
    if (nHashes <= 0)
        return false;

    nonces = {};
    services.nonceProvider.generateNonces(nHashes, nonces);
    return true;
}

void AroMiner::launchGPUTask() {
    uploadInputs_Async();
    run_Async();
    fetchResults_Async();
}

void AroMiner::encode(void *res, size_t reslen, std::string &out) const {
    std::stringstream ss;
    ss << "$argon2i";

    ss << "$v=";
    ss << ARGON_VERSION;

    ss << "$m=";
    ss << argon_params->getMemoryCost();
    ss << ",t=";
    ss << argon_params->getTimeCost();
    ss << ",p=";
    ss << argon_params->getLanes();

    ss << "$";
    char salt[32];
    const char *saltRaw = (const char *)argon_params->getSalt();
    to_base64_(salt, 32, saltRaw, argon_params->getSaltLength());
    ss << salt;

    ss << "$";
    char hash[512];
    to_base64_(hash, 512, res, reslen);
    ss << hash;
    out = ss.str();
}

static std::string extractDuration(const std::string & result) {
    auto sha = SHA512((unsigned char*)result.c_str(), result.size(), nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);

    std::stringstream x;
    x << std::hex;
    x << std::dec << (int)sha[10];
    x << std::dec << (int)sha[15];
    x << std::dec << (int)sha[20];
    x << std::dec << (int)sha[23];
    x << std::dec << (int)sha[31];
    x << std::dec << (int)sha[40];
    x << std::dec << (int)sha[45];
    x << std::dec << (int)sha[55];

    std::string duration = x.str();
    duration.erase(0, std::min(duration.find_first_not_of('0'), duration.size() - 1));

    return duration;
}

AroMiner::ProcessedResults AroMiner::processResults() {
    PerfScope p("AroMiner::processResults()");

    updateNonceProvider();

    auto blockType = nonces.blockDesc.type;
    if (blockType != services.nonceProvider.currentBlockType())
        return ProcessedResults{ false, 0, 0 };

    uint32_t nGood = 0, nHashes = 0;
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        size_t batchSize = memConfig.batchSizes[blockType][i];
        for (size_t j = 0; j < batchSize; ++j) {
            uint8_t* resultPtr = resultsPtr() + 
                nHashes * (argon_params->getLanes() * ARGON2_BLOCK_SIZE);
            uint8_t buffer[32];
            this->argon_params->finalize(buffer, resultPtr);

            std::string encodedArgon;
            encode(buffer, 32, encodedArgon);

            mpz_class mpz_result;
            std::string resultStr = nonces.bases[nHashes] + encodedArgon;
            mpz_result.set_str(extractDuration(resultStr), 10);

            IAroResultsProcessor::Result r {
                services.nonceProvider.salt(blockType),
                nonces.bases[nHashes],
                nonces.nonces[nHashes],
                encodedArgon,
                mpz_result
            };

            nGood += (int)
                services.resultProcessor.processResult({ r, nonces.blockDesc });
            nHashes++;
        }
    }
    return ProcessedResults{ true, nHashes, nGood };
}

std::string AroMiner::describeKernel(BLOCK_TYPE bt) const {
    auto mode = optimizationMode(bt);
    if (bt == BLOCK_CPU) {
        if (mode == argon2::PRECOMPUTE_LOCAL_STATE)
            return "index_local";
        else if (mode == argon2::PRECOMPUTE_SHUFFLE)
            return "index_shuffle";
    }
    else if (mode == argon2::BASELINE)
        return "shuffle";
    return "unknown";
}

std::string AroMiner::describe() const {
    std::ostringstream oss;
    auto describeMiner = [&](BLOCK_TYPE bt) {
        auto &pSizes = memConfig.batchSizes[bt];
        if (pSizes[0] == 0) {
            return "error, batch size <= 0, did you set -b value correctly ?";
        }
        oss << describeKernel(bt) << "(";
        for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
            if (i != 0 && pSizes[i])
                oss << " ";
            if (pSizes[i])
                oss << pSizes[i];
            else
                break;
        }
        oss << ")";
    };
    describeMiner(BLOCK_GPU);
    oss << " ";
    describeMiner(BLOCK_CPU);
    return oss.str();
}
