//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_MINER_H
#define ARIONUM_GPU_MINER_MINER_H

#include <cpprest/http_client.h>
#include <iostream>
#include <gmp.h>
#ifdef _WIN32
#include <mpirxx.h>
#else
#include <gmpxx.h>
#endif
#include <argon2-gpu-common/argon2-common.h>
#include <argon2-gpu-common/argon2params.h>

#include <locale>
#include <codecvt>
#include <string>

// enabling this will always use same pass/salt/nonce and exit(1) if result not matching reference
// (of course will also not submit anything)
#define TEST_OFF (0)
#define TEST_GPU (1)
#define TEST_CPU (2)
#define TEST_MODE (TEST_OFF)

#include "stats.h"
#include "minersettings.h"
#include "minerdata.h"
#include "updater.h"

#define EQ(x, y) ((((0U - ((unsigned)(x) ^ (unsigned)(y))) >> 8) & 0xFF) ^ 0xFF)
#define GT(x, y) ((((unsigned)(y) - (unsigned)(x)) >> 8) & 0xFF)
#define GE(x, y) (GT(y, x) ^ 0xFF)
#define LT(x, y) GT(y, x)
#define LE(x, y) GE(y, x)

using namespace web;
using namespace web::http;
using namespace web::http::client;

const size_t ARGON_OUTLEN = 32;
const size_t ARGON_SALTLEN = 16;

class Miner {
private:
    //const char *alphanum = "0123456789!@#$%^&*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
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
    http_client *client;

    Updater *updater;

    std::vector<std::string> nonces;
    std::vector<std::string> bases;
    std::vector<std::string> argons;
    char *nonceBase64 = new char[64];
    uint8_t *byteBuffer = new uint8_t[32];

    string salt;

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
    explicit Miner(Stats *s, MinerSettings *ms, uint32_t bs, Updater *u) :
        stats(s),
        settings(ms),
        batchSize(bs),
        initial_batchSize(bs),
        rest(0),
        diff(1),
        result(0),
        ZERO(0),
        BLOCK_LIMIT(240),
        limit(0),
        updater(u),
        params(nullptr),
        cpu_batchSize(1)
    {
        http_client_config config;
        utility::seconds timeout(2);
        config.set_timeout(timeout);

        utility::string_t poolAddress = toUtilityString(*ms->getPoolAddress());

        client = new http_client(poolAddress, config);
        generator = std::mt19937(device());
        distribution = std::uniform_int_distribution<int>(0, 255);
        
#if (TEST_MODE == TEST_CPU)
        salt = "0KVwsNr6yT42uDX9"; // == from_base64("MEtWd3NOcjZ5VDQydURYOQ")
#elif (TEST_MODE == TEST_GPU)
        salt = "cifE2rK4nvmbVgQu"; // == from_base64("Y2lmRTJySzRudm1iVmdRdQ")
#else
        salt = randomStr(16);
#endif

        // prepare array of gpu task results buffers
        auto count = batchSize;
        resultBuffers.resize(count);
    };

    virtual void reconfigureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t batchSize) = 0;

    void to_base64(char *dst, size_t dst_len, const void *src, size_t src_len);

    void generateBytes(char *dst, size_t dst_len, uint8_t *buffer, size_t buffer_size);

    void buildBatch();

    void hostPrepareTaskData();
    void hostProcessResults();

    virtual void deviceUploadTaskDataAsync() = 0;
    virtual void deviceLaunchTaskAsync() = 0;
    virtual void deviceFetchTaskResultAsync() = 0;
    virtual void deviceWaitForResults() = 0;
    virtual bool deviceResultsReady() = 0;

    void checkArgon(string *base, string *argon, string *nonce);

    void submit(string *argon, string *nonce, bool d);

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
};


#endif //ARIONUM_GPU_MINER_MINER_H
