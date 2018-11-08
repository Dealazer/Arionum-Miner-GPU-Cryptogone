#pragma once

#include "minerdata.h"

#include <gmp.h>
#ifdef _WIN32
#include <mpirxx.h>
#else
#include <gmpxx.h>
#endif

#include <string>
#include <vector>

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
    virtual const std::string& salt(BLOCK_TYPE bt) const = 0;
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

class IAroResultsProcessor {
public:
    struct Result {
        const std::string &salt;
        const std::string &base;
        const std::string &nonce;
        const std::string &encodedArgon;
        mpz_class &mpz_result;
    };
    struct Input {
        const Result & result;
        const BlockDesc& blockDesc;
    };

    virtual ~IAroResultsProcessor() {};
    virtual bool processResult(const Input& i) = 0;
};