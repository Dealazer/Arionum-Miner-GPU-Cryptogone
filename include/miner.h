//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINER_H
#define ARIONUM_GPU_MINER_MINER_H

#include "mining_services.h"

#include <argon2-gpu-common/argon2-common.h>
#include <argon2-gpu-common/argon2params.h>

#include <memory>

const argon2::Type ARGON_TYPE = argon2::ARGON2_I;
const argon2::Version ARGON_VERSION = argon2::ARGON2_VERSION_13;

class IGPUMiner {
public:
    virtual ~IGPUMiner() {};
    virtual bool resultsReady() = 0;
    virtual argon2::time_point asyncStartTime() const = 0;

protected:
    virtual void reconfigureKernel() = 0;
    virtual void uploadInputs_Async() = 0;
    virtual void fetchResults_Async() = 0;
    virtual void run_Async() = 0;
    virtual uint8_t * resultsPtr() = 0;
};

class AroMiner : public IGPUMiner {
public:
    struct Services {
        IAroNonceProvider & nonceProvider;
        IAroResultsProcessor & resultProcessor;
    };

    struct ProcessedResults {
        bool valid;
        uint32_t nHashes, nGood;
    };

    AroMiner(const argon2::MemConfig &memConfig, const Services& services,
        argon2::OPT_MODE cpuBlocksOptimizationMode);
    
    bool updateNonceProvider();
    bool generateNonces();
    void launchGPUTask();
    ProcessedResults processResults();

    uint32_t nHashesPerRun(BLOCK_TYPE bt) const;

    BLOCK_TYPE taskBlockType() const 
    { return nonces.blockDesc.type; };
    BLOCK_TYPE providerBlockType() const 
    { return services.nonceProvider.currentBlockType(); };
    std::string describe() const;

protected:
    argon2::OptParams configureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes);
    
    void encode(void *res, size_t reslen, std::string &out) const;
    bool needReconfigure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes) const;
    argon2::OPT_MODE optimizationMode(BLOCK_TYPE blockType) const;
    std::string describeKernel(BLOCK_TYPE bt) const;

    Services services;
    argon2::MemConfig memConfig;
    std::unique_ptr<argon2::Argon2Params> argon_params;
    argon2::OptParams optPrms;
    argon2::Argon2iMiningConfig miningConfig;
    Nonces nonces;
    argon2::OPT_MODE cpuBlocksOptimizationMode;
};

#endif //ARIONUM_GPU_MINER_MINER_H
