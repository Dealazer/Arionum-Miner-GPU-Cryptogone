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

const size_t ARGON_OUTLEN = 32;
const size_t ARGON_SALTLEN = 16;


class Stats;
class Updater;

class Miner {
public:
    static const int MAX_BLOCKS_BUFFERS = 4;

protected:
    argon2::Type type = argon2::ARGON2_I;
    argon2::Version version = argon2::ARGON2_VERSION_13;
    argon2::Argon2Params *params;
    
    MinerData data;
    MinerSettings settings;
    Updater *updater;
    Stats *stats;

private:
    std::random_device device;
    std::mt19937 generator;
    std::uniform_int_distribution<int> distribution;
    const char *alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    char genRandom(int v);

    std::string salt;
    char nonceBase64[64];
    uint8_t byteBuffer[32];
    std::vector<std::string> nonces[MAX_BLOCKS_BUFFERS];
    std::vector<std::string> bases[MAX_BLOCKS_BUFFERS];

    mpz_class ZERO;
    mpz_class BLOCK_LIMIT;
    mpz_class rest;
    mpz_class result;
    mpz_class diff;
    mpz_class limit;

    web::http::client::http_client *client;

    size_t maxMemUsage;

    std::vector<uint8_t*> resultsPtrs[MAX_BLOCKS_BUFFERS];

protected:
    argon2::t_optParams configureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes);
    argon2::t_optParams precomputeArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes);
    bool needReconfigure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes);
    void to_base64(char *dst, size_t dst_len, const void *src, size_t src_len);
    void generateBytes(char *dst, size_t dst_len, uint8_t *buffer, size_t buffer_size);
    void buildBatch();
    bool checkArgon(std::string *base, std::string *argon, std::string *nonce);
    void submit(std::string *argon, std::string *nonce, bool d, bool isBlock);
    void submitReject(std::string msg, bool isBlock);
    void encode(void *res, size_t reslen, std::string &out);
    std::string randomStr(int length);

public:
    Miner(size_t maxMemUsage, Stats *s, MinerSettings &ms, Updater *u);

    void hostPrepareTaskData();
    bool hostProcessResults();
    bool canMineBlock(BLOCK_TYPE type);
    BLOCK_TYPE getCurrentBlockType();

public:
    virtual void deviceUploadTaskDataAsync() = 0;
    virtual void deviceLaunchTaskAsync() = 0;
    virtual void deviceFetchTaskResultAsync() = 0;
    virtual void deviceWaitForResults() = 0;
    virtual bool deviceResultsReady() = 0;
    virtual void reconfigureArgon(
        uint32_t t_cost, uint32_t m_cost, uint32_t lanes, 
        uint32_t batchSize) = 0;

public:
    static uint32_t getMemCost(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 524288 : 16384;
    }

    static uint32_t getPasses(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 1 : 4;
    }

    static uint32_t getLanes(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 1 : 4;
    }

    uint32_t getNbHashesPerIteration();

    // in progress
public:
    struct MemConfig {
        size_t batchSizes[BLOCK_TYPE_COUNT][MAX_BLOCKS_BUFFERS];
        size_t blocksBuffers[MAX_BLOCKS_BUFFERS];
        size_t index;
        size_t in;
        size_t out;
    };
    bool initialize(size_t maxMemUsage);

public:
    virtual MemConfig configure(size_t maxMemUsage) = 0;
    virtual bool initialize(MemConfig &mcfg) = 0;

protected:
    MemConfig memConfig;

    // to review
public:
    //uint32_t getInitialBatchSize() const { return initial_batchSize; };
    //uint32_t getCurrentBatchSize() const { return batchSize; };
    //uint32_t getCPUBatchSize() const { return cpu_batchSize; };
    //virtual size_t getMemoryUsage() const = 0;
    //virtual size_t getMemoryUsedPerBatch() const = 0;
    std::string getInfo() const;
    //void computeCPUBatchSize();
};


#endif //ARIONUM_GPU_MINER_MINER_H
