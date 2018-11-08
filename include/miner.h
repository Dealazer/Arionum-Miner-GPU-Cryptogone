//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINER_H
#define ARIONUM_GPU_MINER_MINER_H

#include "minerdata.h"
#include "minersettings.h"

#include <gmp.h>
#ifdef _WIN32
#include <mpirxx.h>
#else
#include <gmpxx.h>
#endif
#include <cpprest/http_client.h>
#include <argon2-gpu-common/argon2params.h>

#include <string>
#include <vector>

#define EQ(x, y) ((((0U - ((unsigned)(x) ^ (unsigned)(y))) >> 8) & 0xFF) ^ 0xFF)
#define GT(x, y) ((((unsigned)(y) - (unsigned)(x)) >> 8) & 0xFF)
#define GE(x, y) (GT(y, x) ^ 0xFF)
#define LT(x, y) GT(y, x)
#define LE(x, y) GE(y, x)

const argon2::Type ARGON_TYPE = argon2::ARGON2_I;
const argon2::Version ARGON_VERSION = argon2::ARGON2_VERSION_13;

const size_t ARGON_OUTLEN = 32;
const size_t ARGON_SALTLEN = 16;

class Stats;
class Updater;

class RandomBytesGenerator {
protected:
    RandomBytesGenerator();
    void generateBytes(char *dst, std::size_t dst_len,
        std::uint8_t *buffer, std::size_t buffer_size);

private:
    std::random_device rdevice;
    std::mt19937 generator;
    std::uniform_int_distribution<int> distribution;
};

struct BlockDesc {
    BlockDesc() : 
        mpz_diff(0), mpz_limit(0), public_key(), type(BLOCK_MAX) {};

    mpz_class mpz_diff;
    mpz_class mpz_limit;
    std::string public_key;
    BLOCK_TYPE type;
};

struct Nonces {
    BlockDesc blockDesc{};
    std::vector<std::string> nonces{};
    std::vector<std::string> bases{};

    void prepareAdd(BlockDesc &new_blockDesc, std::size_t count) {
        if (blockDesc.type != new_blockDesc.type) {
            nonces.clear();
            bases.clear();
        }
        blockDesc = new_blockDesc;
        auto n = nonces.size() + count;
        nonces.reserve(n);
        bases.reserve(n);
    }
};

class IAroNonceProvider {
public:
    virtual ~IAroNonceProvider() {};
    virtual const string& salt(BLOCK_TYPE bt) const = 0;
    virtual bool update() = 0;
    virtual BLOCK_TYPE currentBlockType() const = 0;
    virtual BlockDesc currentBlockDesc() const = 0;
    virtual void generateNonces(std::size_t count, Nonces &nonces) = 0;

protected:
    virtual void generateNoncesImpl(std::size_t count, Nonces &nonces) = 0;
};

class AroNonceProvider : public IAroNonceProvider {
public:
    void generateNonces(std::size_t count, Nonces &nonces) override {
        auto new_blockDesc = currentBlockDesc();
        nonces.prepareAdd(new_blockDesc, count);
        generateNoncesImpl(count, nonces);
    };
};

class AroNonceProviderPool : public AroNonceProvider,
    public RandomBytesGenerator {
public:
    AroNonceProviderPool(Updater & updater);
    bool update() override;
    const string& salt(BLOCK_TYPE bt) const override;
    BLOCK_TYPE currentBlockType() const override { return bd.type; };
    BlockDesc currentBlockDesc() const override { return bd; };

protected:
    void generateNoncesImpl(std::size_t count, Nonces &nonces) override;

private:
    Updater & updater;
    MinerData data;
    char nonceBase64[64];
    uint8_t byteBuffer[32];
    std::string initSalt;
    BlockDesc bd;
};

class AroNonceProviderTestMode : public AroNonceProvider {
public:
    AroNonceProviderTestMode(Stats & stats) : stats(stats) {};
    const string& salt(BLOCK_TYPE bt) const override;
    bool update() override;
    BLOCK_TYPE currentBlockType() const override { return blockType; };
    BlockDesc currentBlockDesc() const override {
        BlockDesc bd;
        bd.type = blockType;
        return bd;
    };

protected:
    void generateNoncesImpl(std::size_t count, Nonces &nonces) override;

private:
    BLOCK_TYPE blockType;
    Stats & stats;
};

class IAroResultsProcessor {
public:
    struct Result {
        const std::string &salt;
        const std::string &base;
        const std::string &nonce;
        const std::string &encodedArgon;
    };
    struct Input {
        const Result & result;
        const BlockDesc& blockDesc;
    };

    virtual ~IAroResultsProcessor() {};
    virtual bool processResult(const Input& i) = 0;
};

class AroResultsProcessorPool : public IAroResultsProcessor {
public:
    AroResultsProcessorPool(const MinerSettings & ms, Stats & stats);
    virtual bool processResult(const Input& i) override;

private:
    struct SubmitParams {
        std::string nonce;
        const std::string &argon;
        const std::string &public_key;
        bool d;
        bool isBlock;
    };
    void submit(SubmitParams & prms);
    void submitReject(const string &msg, bool isBlock);

    MinerSettings settings;
    Stats & stats;

    const std::size_t SUBMIT_HTTP_TIMEOUT_SECONDS = 2;
    std::unique_ptr<web::http::client::http_client> client;

    mpz_class BLOCK_LIMIT, mpz_ZERO, mpz_result, mpz_rest;
};

class AroResultsProcessorTestMode : public IAroResultsProcessor {
public:
    virtual bool processResult(const Input& i) override;
};

class IGPUMiner {
public:
    virtual ~IGPUMiner() {};
    virtual bool resultsReady() = 0;
    virtual void waitResults() = 0;

protected:
    virtual void reconfigureKernel() = 0;
    virtual void uploadInputs_Async() = 0;
    virtual void fetchResults_Async() = 0;
    virtual void run_Async() = 0;
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

    uint32_t nHashesPerRun() const;
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
    std::vector<uint8_t*> resultsPtrs[MAX_BLOCKS_BUFFERS];
    Nonces nonces;
    argon2::OPT_MODE cpuBlocksOptimizationMode;
};

#endif //ARIONUM_GPU_MINER_MINER_H
