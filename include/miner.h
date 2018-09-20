//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINER_H
#define ARIONUM_GPU_MINER_MINER_H

#include "minerdata.h"

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

class MinerSettings;
class Stats;
class Updater;

class Miner {
private:
    const char *alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    char genRandom(int v) {
        return alphanum[v];
    }

    std::string randomStr(int length) {
        size_t stringLength = strlen(alphanum) - 1;
        std::stringstream ss;
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator
        std::uniform_int_distribution<> distr(0, (int)stringLength); // define the range

        for (int i = 0; i < length; ++i) {
            ss << genRandom(distr(eng));
        }
        return ss.str();
    }

protected:
    mpz_class ZERO;
    mpz_class BLOCK_LIMIT;
    mpz_class rest;
    mpz_class result;
    mpz_class diff;
    mpz_class limit;

    Stats *stats;
    MinerSettings *settings;
    uint32_t batchSize;
    uint32_t initial_batchSize;
    uint32_t cpu_batchSize;
    MinerData data;
    web::http::client::http_client *client;

    Updater *updater;

    std::vector<std::string> nonces;
    std::vector<std::string> bases;
    char *nonceBase64 = new char[64];
    uint8_t *byteBuffer = new uint8_t[32];

    std::string salt;

    std::random_device device;
    std::mt19937 generator;
    std::uniform_int_distribution<int> distribution;

    argon2::Type type = argon2::ARGON2_I;
    argon2::Version version = argon2::ARGON2_VERSION_13;
    argon2::Argon2Params *params;

    std::vector<uint8_t*> resultBuffers;

    argon2::t_optParams configure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t batchSize);
    argon2::t_optParams precompute(uint32_t t_cost, uint32_t m_cost, uint32_t lanes);
    bool needReconfigure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t newBatchSize);

public:
    explicit Miner(Stats *s, MinerSettings *ms, uint32_t bs, Updater *u);

    virtual void reconfigureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t batchSize) = 0;

    void to_base64(char *dst, size_t dst_len, const void *src, size_t src_len);

    void generateBytes(char *dst, size_t dst_len, uint8_t *buffer, size_t buffer_size);

    void buildBatch();

    void hostPrepareTaskData();
    bool hostProcessResults();

    virtual void deviceUploadTaskDataAsync() = 0;
    virtual void deviceLaunchTaskAsync() = 0;
    virtual void deviceFetchTaskResultAsync() = 0;
    virtual void deviceWaitForResults() = 0;
    virtual bool deviceResultsReady() = 0;

    bool checkArgon(std::string *base, std::string *argon, std::string *nonce);

    void submit(std::string *argon, std::string *nonce, bool d, bool isBlock);
    void submitReject(std::string msg, bool isBlock);

    void encode(void *res, size_t reslen, std::string &out);

    uint32_t getInitialBatchSize() const { return initial_batchSize; };
    uint32_t getCurrentBatchSize() const { return batchSize; };
    uint32_t getCPUBatchSize() const { return cpu_batchSize; };

    virtual size_t getMemoryUsage() const = 0;
    virtual size_t getMemoryUsedPerBatch() const = 0;
    std::string getInfo() const;

    static uint32_t getMemCost(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 524288 : 16384;
    }

    static uint32_t getPasses(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 1 : 4;
    }

    static uint32_t getLanes(BLOCK_TYPE type) {
        return (type == BLOCK_CPU) ? 1 : 4;
    }

    bool mineBlock(BLOCK_TYPE type);

    void computeCPUBatchSize();

    BLOCK_TYPE getCurrentBlockType() {
        return data.getBlockType();
    }
};


#endif //ARIONUM_GPU_MINER_MINER_H
