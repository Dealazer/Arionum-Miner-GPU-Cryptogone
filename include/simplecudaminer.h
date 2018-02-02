//
// Created by guli on 02/02/18.
//

#ifndef ARIONUM_GPU_MINER_SIMPLECUDAMINER_H
#define ARIONUM_GPU_MINER_SIMPLECUDAMINER_H

#include <gmp.h>
#include "minerdata.h"
#include "stats.h"
#include "minersettings.h"
#include <argon2-cuda/programcontext.h>
#include <argon2-cuda/processingunit.h>
#include <argon2-cuda/globalcontext.h>
#include <cpprest/http_client.h>
#include <cpprest/json.h>


#define EQ(x, y) ((((0U - ((unsigned)(x) ^ (unsigned)(y))) >> 8) & 0xFF) ^ 0xFF)
#define GT(x, y) ((((unsigned)(y) - (unsigned)(x)) >> 8) & 0xFF)
#define GE(x, y) (GT(y, x) ^ 0xFF)
#define LT(x, y) GT(y, x)
#define LE(x, y) GE(y, x)

using namespace web;
using namespace web::http;
using namespace web::http::client;
using namespace argon2;
using namespace std;

class SimpleCudaMiner {
private:

    mpz_t ZERO;
    mpz_t BLOCK_LIMIT;
    unique_ptr<char[]> AS_SHA = std::unique_ptr<char[]>(new char[128]);
    http_client client;
    int batchSize;
    size_t deviceIndex;
    MinerData updateData;
    Stats stats;
    MinerSettings minerSettings;
    cuda::ProgramContext *progCtx;
    argon2::Argon2Params *params;

    char *alphanum;

    int stringLength;

    char genRandom(int v);

    string randomStr(int length);

    void parseUpdateJson(const json::value &jvalue);

    int b64_byte_to_char(unsigned x);

    size_t to_base64(char *dst, size_t dst_len, const void *src, size_t src_len);

    void generateBytes(char *dst, size_t dst_len);

    void buildBatch(std::vector<std::string> *nonces, std::vector<std::string> *bases, int batchSize);

    char *encode(argon2::cuda::ProgramContext *progCtx, argon2::Argon2Params *params, void *res, size_t reslen);

    void computeHash(argon2::cuda::ProcessingUnit *unit, argon2::cuda::ProgramContext *progCtx,
                     argon2::Argon2Params *params, std::vector<std::string> *bases, std::vector<std::string> *argons,
                     int batchSize);

    void updateInfoRequest(http_client &client);

    void submit(string *argon, string *nonce);

    void toHex(unsigned char *sha);

    void checkArgon(string *base, string *argon, string *nonce);

public:

    SimpleCudaMiner(MinerSettings ms, size_t di);

    void start();

};

#endif //ARIONUM_GPU_MINER_SIMPLECUDAMINER_H
