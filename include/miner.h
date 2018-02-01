//
// Created by guli on 31/01/18.
//

#ifndef ARIONUM_GPU_MINER_MINER_H
#define ARIONUM_GPU_MINER_MINER_H

#include <cpprest/http_client.h>
#include <iostream>
#include <gmp.h>
#include <gmpxx.h>
#include <argon2-gpu-common/argon2-common.h>
#include <argon2-gpu-common/argon2params.h>

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

class Miner {
protected:
    mpz_class ZERO;
    mpz_class BLOCK_LIMIT;
    mpz_class rest;
    mpz_class result;
    mpz_class diff;
    mpz_class limit;

    Stats *stats;
    MinerSettings *settings;
    MinerData *data;
    http_client *client;

    Updater *updater;

    std::vector<std::string> nonces;
    std::vector<std::string> bases;
    std::vector<std::string> argons;
    char *nonceBase64 = new char[64];
    uint8_t *byteBuffer = new uint8_t[32];

    char *saltBase64 = new char[32];
    uint8_t *saltBuffer = new uint8_t[16];

    std::random_device device;
    std::mt19937 generator;
    std::uniform_int_distribution<uint8_t> distribution;

    argon2::Type type = argon2::ARGON2_I;
    argon2::Version version = argon2::ARGON2_VERSION_13;
    argon2::Argon2Params *params;

public:
    explicit Miner(Stats *s, MinerSettings *ms, MinerData *d, Updater *u) : stats(s),
                                                                                  settings(ms),
                                                                                  rest(0),
                                                                                  diff(1),
                                                                                  result(0),
                                                                                  ZERO(0),
                                                                                  BLOCK_LIMIT(240),
                                                                                  limit(0),
                                                                                  updater(u) {
        data = d->getCopy();
        client = new http_client(U(ms->getPoolAddress()->c_str()));
        generator = std::mt19937(device());
        distribution = std::uniform_int_distribution<uint8_t>(0, 255);
    };

    void mine();

    void to_base64(char *dst, size_t dst_len, const void *src, size_t src_len);

    void generateBytes(char *dst, size_t dst_len, uint8_t *buffer, size_t buffer_size);

    void buildBatch();

    virtual void computeHash() = 0;

    void checkArgon(string *base, string *argon, string *nonce);

    void submit(string *argon, string *nonce);

    char *encode(void *res, size_t reslen);


};

#endif //ARIONUM_GPU_MINER_MINER_H
