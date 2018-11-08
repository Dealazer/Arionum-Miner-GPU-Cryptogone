#pragma once

#include "mining_services.h"
#include "minersettings.h"

#include <cpprest/http_client.h>

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

    mpz_class BLOCK_LIMIT, mpz_ZERO, mpz_rest;
};
