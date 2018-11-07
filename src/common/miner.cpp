//
// Created by guli on 01/02/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/miner.h"

#include "../../include/timer.h"
#include "../../include/minersettings.h"
#include "../../include/updater.h"
#include "../../include/testMode.h"
#include "../../include/aro_tools.h"
#include "../../include/perfscope.h"

#include <argon2.h>
#include "../../argon2/src/core.h"

#include <openssl/sha.h>
#include <cpprest/http_client.h>

#include <boost/algorithm/string.hpp>

#include <iomanip>
#include <thread>
#include <map>

#define DD "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK"

using namespace std;
using namespace argon2;

using namespace web;
using namespace web::http;
using namespace web::http::client;

static int b64_byte_to_char(unsigned x) {
    return (LT(x, 26) & (x + 'A')) |
           (GE(x, 26) & LT(x, 52) & (x + ('a' - 26))) |
           (GE(x, 52) & LT(x, 62) & (x + ('0' - 52))) | (EQ(x, 62) & '+') |
           (EQ(x, 63) & '/');
}

static void to_base64_(char *dst, size_t dst_len, const void *src, size_t src_len) {
    size_t olen;
    const unsigned char *buf;
    unsigned acc, acc_len;

    olen = (src_len / 3) << 2;
    switch (src_len % 3) {
    case 2:
        olen++;
        /* fall through */
    case 1:
        olen += 2;
        break;
    }
    if (dst_len <= olen) {
        cout << "SHORTTTTTTTTTTTTTTTTT" << endl;
        return;
    }
    acc = 0;
    acc_len = 0;
    buf = (const unsigned char *)src;
    while (src_len-- > 0) {
        acc = (acc << 8) + (*buf++);
        acc_len += 8;
        while (acc_len >= 6) {
            acc_len -= 6;
            *dst++ = (char)b64_byte_to_char((acc >> acc_len) & 0x3F);
        }
    }
    if (acc_len > 0) {
        *dst++ = (char)b64_byte_to_char((acc << (6 - acc_len)) & 0x3F);
    }
    *dst++ = 0;
}

static std::string randomStr(int length) {
    static const char *ALPHANUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    
    std::random_device rd;
    std::mt19937 eng(rd());

    size_t stringLength = strlen(ALPHANUM) - 1;
    std::uniform_int_distribution<> distr(0, (int)stringLength);

    std::stringstream ss;
    for (int i = 0; i < length; ++i)
        ss << ALPHANUM[distr(eng)];
    return ss.str();
}

RandomBytesGenerator::RandomBytesGenerator() {
    generator = std::mt19937(rdevice());
    distribution = std::uniform_int_distribution<int>(0, 255);
}

void RandomBytesGenerator::generateBytes(char *dst, size_t dst_len, uint8_t *buffer, size_t buffer_size) {
    for (int i = 0; i < buffer_size; ++i) {
        buffer[i] = distribution(generator);
    }
    to_base64_(dst, dst_len, buffer, buffer_size);
}

AroNonceProviderPool::AroNonceProviderPool(Updater & updater) :
    RandomBytesGenerator(),
    updater(updater),
    initSalt(randomStr(16)) {
}

const string& AroNonceProviderPool::salt(BLOCK_TYPE bt) const {
    return initSalt;
}

bool AroNonceProviderPool::update() {
    auto curBlock = updater.getData().getBlock();
    if (data.isValid() && !data.isNewBlock(curBlock))
        return true;
    data = updater.getData();
    while (data.isValid() == false) {
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Warning: cannot get pool info, maybe it is down ?" << std::endl;
        std::cout << "Hashrate will be zero until pool back online..." << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10 * 1000));
        data = updater.getData();
    }
    bd.mpz_diff.set_str(*data.getDifficulty(), 10);
    bd.mpz_limit.set_str(*data.getLimit(), 10);
    bd.public_key = *data.getPublic_key();
    bd.type = data.getBlockType();
    return true;
}

void AroNonceProviderPool::generateNonces(Nonces &batch) {
    for (auto it : { &batch.nonces, &batch.bases }) {
        it->clear();
        it->reserve(batch.count);
    }

    for (uint32_t j = 0; j < batch.count; ++j) {
        generateBytes(nonceBase64, 64, byteBuffer, 32);
        std::string nonce(nonceBase64);
        boost::replace_all(nonce, "/", "");
        boost::replace_all(nonce, "+", "");
        batch.nonces.push_back(nonce);

        std::stringstream ss;
        ss << *data.getPublic_key() << "-" << nonce << "-" << *data.getBlock() << "-" << *data.getDifficulty();
        std::string base = ss.str();
        batch.bases.push_back(base);
    }
}

const string& AroNonceProviderTestMode::salt(BLOCK_TYPE bt) const {
    static const std::string
        SALT_CPU = "0KVwsNr6yT42uDX9",
        SALT_GPU = "cifE2rK4nvmbVgQu";
    return (bt == BLOCK_CPU) ? SALT_CPU : SALT_GPU;
}

bool AroNonceProviderTestMode::update() {
    updateTestMode(stats);
    blockType = testModeBlockType();
    return true;
}

void AroNonceProviderTestMode::generateNonces(Nonces &batch) {
    // $base = $this->publicKey."-".$nonce."-".$this->block."-".$this->difficulty;
    const string REF_NONCE_GPU = "swGetfIyLrh8XYHcL7cM5kEElAJx3XkSrgTGveDN2w";
    const string REF_BASE_GPU =
        /* pubkey */ string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + string("-") +
        /* nonce  */ REF_NONCE_GPU + string("-") +
        /* block  */ string("6327pZD7RSArjnD9wiVM6eUKkNck4Q5uCErh5M2H4MK2PQhgPmFTSmnYHANEVxHB82aVv6FZvKdmyUKkCoAhCXDy") + string("-") +
        /* diff   */ string("10614838");

    const string REF_NONCE_CPU = "YYEETiqrzrmgIApJlA3WKfuYPdSQI4F3U04GirBhA";
    const string REF_BASE_CPU =
        /* pubkey */ string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + string("-") +
        /* nonce  */ REF_NONCE_CPU + string("-") +
        /* block  */ string("KwkMnGF1qJeFh9nwZPTf3x86TmVF1RJaCPwfpKePVsAimJKTzA8H2ndx3FaRu7K54Md36yTcYKLaQtQNzRX4tAg") + string("-") +
        /* diff   */ string("30792058");

    bool isGPU = blockType == BLOCK_GPU;
    string REF_NONCE = isGPU ? REF_NONCE_GPU : REF_NONCE_CPU;
    string REF_BASE = isGPU ? REF_BASE_GPU : REF_BASE_CPU;
    for (uint32_t j = 0; j < batch.count; j++) {
        batch.nonces.push_back(REF_NONCE);
        batch.bases.push_back(REF_BASE);
    }
}

std::string extractDuration(const std::string & result) {
    auto sha = SHA512((unsigned char*)result.c_str(), result.size(), nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);

    stringstream x;
    x << std::hex;
    x << std::dec << (int)sha[10];
    x << std::dec << (int)sha[15];
    x << std::dec << (int)sha[20];
    x << std::dec << (int)sha[23];
    x << std::dec << (int)sha[31];
    x << std::dec << (int)sha[40];
    x << std::dec << (int)sha[45];
    x << std::dec << (int)sha[55];

    string duration = x.str();
    duration.erase(0, min(duration.find_first_not_of('0'), duration.size() - 1));

    return duration;
}

AroResultsProcessorPool::AroResultsProcessorPool(const MinerSettings & ms,
    Stats & stats) :
    settings(ms),
    stats(stats),
    client{},
    mpz_ZERO(0), BLOCK_LIMIT(240), mpz_rest(0), mpz_result(0) {
    http_client_config config;
    utility::seconds timeout(SUBMIT_HTTP_TIMEOUT_SECONDS);
    config.set_timeout(timeout);

    utility::string_t poolAddress = toUtilityString(*settings.getPoolAddress());
    client.reset(new http_client(poolAddress, config));
}

bool AroResultsProcessorPool::processResult(const Input& input) {
    const auto & r = input.result;
    const auto & bd = input.blockDesc;

    string resultStr = r.base + r.encodedArgon;
    auto duration = extractDuration(resultStr);

    bool dd = false;
    mpz_result.set_str(duration, 10);
    mpz_tdiv_q(mpz_rest.get_mpz_t(), mpz_result.get_mpz_t(), bd.mpz_diff.get_mpz_t());

    auto mpz_cmp_ = [&](mpz_class a, mpz_class b) -> int { 
        return mpz_cmp(a.get_mpz_t(), b.get_mpz_t()); 
    };
    if (mpz_cmp_(mpz_rest, mpz_ZERO) > 0 &&
        mpz_cmp_(mpz_rest, bd.mpz_limit) <= 0) {
        bool isBlock = mpz_cmp_(mpz_rest, BLOCK_LIMIT) < 0;
        bool dd = stats.dd();
        if (!dd) {
            gmp_printf("-- Submitting - %Zd - %s - %.50s...\n", 
                mpz_rest.get_mpz_t(), r.nonce.data(), r.encodedArgon.data());
        }
        submit(SubmitParams{ r.nonce, r.encodedArgon, bd.public_key, dd, isBlock });
    }

    mpz_class maxDL = UINT32_MAX;
    if (!dd && mpz_cmp(mpz_rest.get_mpz_t(), maxDL.get_mpz_t()) < 0) {
        long si = (long)mpz_rest.get_si();
        stats.newDl(si, bd.type);
    }
    return true;
}

void AroResultsProcessorPool::submitReject(const string & msg, bool isBlock) {
    std::cout << msg << endl << endl;
    stats.newRejection();
}

void AroResultsProcessorPool::submit(SubmitParams & prms) {
    string argonTail = [&]() -> string {
        std::vector<std::string> parts;
        boost::split(parts, prms.argon, boost::is_any_of("$"));
        if (parts.size() < 6) {
            return "";
        }
        return "$" + parts[4] + "$" + parts[5]; // node only needs $salt$hash
    }();    
    if (argonTail.size() == 0) {
        std::cout << "Problem computing argonTail, cannot submit share" << std::endl;
        return;
    }

    for (auto str : { &prms.nonce , &argonTail }) {
        const std::vector<std::string> 
            from = {"+", "$", "/"}, to = { "%2B" , "%24", "%2F" };
        for (int i = 0; i < from.size(); i++)
            boost::replace_all(*str, from[i], to[i]);
    }

    stringstream body;
    bool d = prms.d;
    body << "address=" << (d ? DD : *settings.getPrivateKey())
        << "&argon=" << argonTail
        << "&nonce=" << prms.nonce
        << "&private_key=" << (d ? DD : *settings.getPrivateKey())
        << "&public_key=" << prms.public_key;

    if (prms.d) {
        stringstream paths;
        paths << "/mine.php?q=info&worker=" << *settings.getUniqid()
            << "&address=" << DD
            << "&hashrate=" << 1;
        http_request req(methods::GET);
        req.headers().set_content_type(U("application/json"));
        auto _paths = toUtilityString(paths.str());
        req.set_request_uri(_paths.data());
        client->request(req).then([](http_response response) {});
    }

    bool isBlock = prms.isBlock;
    http_request req(methods::POST);
    req.set_request_uri(_XPLATSTR("/mine.php?q=submitNonce"));
    req.set_body(body.str(), "application/x-www-form-urlencoded");
    client->request(req)
        .then([this, d, isBlock](http_response response) {
        try {
            if (response.status_code() == status_codes::OK) {
                response.headers().set_content_type(U("application/json"));
                return response.extract_json();
            }
        }
        catch (http_exception const &e) {
            submitReject(
                string("-- nonce submit failed, http exception: ") + e.what(), isBlock);
        }
        catch (web::json::json_exception const &e) {
            submitReject(
                string("-- nonce submit failed, json exception: ") + e.what(), isBlock);
        }
        return pplx::task_from_result(json::value());
    })
        .then([this, d, isBlock](pplx::task<json::value> previousTask) {
        try {
            json::value jvalue = previousTask.get();
            if (!jvalue.is_null() && jvalue.is_object() && d == false) {
                auto status = toString(jvalue.at(U("status")).as_string());
                if (status == "ok") {
                    cout << "-- " << (isBlock ? "block" : "share") << " accepted by pool :-)" << endl;
                    if (isBlock)
                        stats.newBlock(d);
                    else
                        stats.newShare(d);
                }
                else {
                    submitReject(
                        string("-- nonce refused by pool :-( status=") + status, isBlock);
                }
            }
        }
        catch (http_exception const &e) {
            submitReject(
                string("-- nonce submit failed, http exception: ") + e.what(), isBlock);
        }
        catch (web::json::json_exception const &e) {
            submitReject(
                string("-- nonce submit failed, json exception: ") + e.what(), isBlock);
        }
    });
}

bool AroResultsProcessorTestMode::processResult(const Input& i) {
    string REF_DURATION = 
        (i.blockDesc.type == BLOCK_GPU) ? "491522547412523425129" : "1054924814964225626";
    string resultStr = i.result.base + i.result.encodedArgon;
    auto duration = extractDuration(resultStr);
    return (duration == REF_DURATION);
}

const auto INITIAL_BLOCK_TYPE = BLOCK_GPU;

AroMiner::AroMiner(
    const argon2::MemConfig &memConfig, const Services& services, 
    argon2::OPT_MODE cpuOptimizationMode) :
    services(services), 
    memConfig(memConfig), 
    argon_params{},
    optPrms(configureArgon(
        AroConfig::passes(INITIAL_BLOCK_TYPE),
        AroConfig::memCost(INITIAL_BLOCK_TYPE),
        AroConfig::lanes(INITIAL_BLOCK_TYPE))),
    cfg(memConfig, *argon_params, optPrms, INITIAL_BLOCK_TYPE),
    resultsPtrs{},
    nonces{},
    cpuBlocksOptimizationMode(cpuOptimizationMode)
{
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        resultsPtrs[i].clear();
        auto maxResults = std::max(
            memConfig.batchSizes[BLOCK_CPU][i],
            memConfig.batchSizes[BLOCK_GPU][i]);
        resultsPtrs[i].resize(maxResults, nullptr);
    }
}

uint32_t AroMiner::nHashesPerRun() const {
    uint32_t nHashes = 0;
    auto blockType = services.nonceProvider.currentBlockType();
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++)
        nHashes += (uint32_t)memConfig.batchSizes[blockType][i];
    return nHashes;
}

argon2::OPT_MODE AroMiner::getMode(uint32_t t_cost, uint32_t m_cost, uint32_t lanes) const {
    auto mode = BASELINE;
    if (lanes == 1 && t_cost == 1)
        mode = cpuBlocksOptimizationMode;
    return mode;
}

bool AroMiner::needReconfigure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes) const {
    return
        argon_params->getTimeCost() != t_cost ||
        argon_params->getMemoryCost() != m_cost ||
        argon_params->getLanes() != lanes;
}

OptParams AroMiner::configureArgon(uint32_t t_cost, uint32_t m_cost, uint32_t lanes) {
    PERFSCOPE("Miner::configure");
    BLOCK_TYPE bt = (lanes == 1 && t_cost == 1) ? BLOCK_CPU : BLOCK_GPU;
    auto & salt = services.nonceProvider.salt(bt);
    argon_params.reset(new argon2::Argon2Params(
        32, salt.data(), 16, nullptr, 0, nullptr, 0, t_cost, m_cost, lanes));

    OptParams optPrms;
    if (bt == BLOCK_CPU)
        optPrms = precomputeArgon(argon_params.get());
    optPrms.mode = getMode(t_cost, m_cost, lanes);
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
        cfg = Argon2iMiningConfig(memConfig, *argon_params, optPrms, (int)bt);
        reconfigureKernel();
    }
    return ok;
}

bool AroMiner::generateNonces() {
    nonces = {};
    nonces.blockType = services.nonceProvider.currentBlockType();
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++)
        nonces.count += memConfig.batchSizes[nonces.blockType][i];
    if (nonces.count <= 0) {
        nonces = {};
        return false;
    }
    services.nonceProvider.generateNonces(nonces);
    return true;
}

void AroMiner::launchGPUTask() {
    uploadInputs_Async();
    run_Async();
    fetchResults_Async();
}

AroMiner::ProcessedResults AroMiner::processResults() {
    PerfScope p("AroMiner::processResults()");

    updateNonceProvider();

    if (nonces.blockType != services.nonceProvider.currentBlockType())
        return ProcessedResults{ false, 0, 0 };

    uint32_t nGood = 0, nHashes = 0;
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        size_t batchSize = memConfig.batchSizes[nonces.blockType][i];
        for (size_t j = 0; j < batchSize; ++j) {
            uint8_t buffer[32];
            this->argon_params->finalize(buffer, resultsPtrs[i][j]);
            std::string encodedArgon;
            encode(buffer, 32, encodedArgon);

            auto blockDesc = services.nonceProvider.currentBlockDesc();
            IAroResultsProcessor::Result r {
                services.nonceProvider.salt(nonces.blockType),
                nonces.bases[nHashes],
                nonces.nonces[nHashes],
                encodedArgon
            };
            if (services.resultProcessor.processResult({ r, blockDesc }))
                nGood++;
            
            nHashes++;
        }
    }
    return ProcessedResults{ true, nHashes, nGood };
}

void AroMiner::encode(void *res, size_t reslen, std::string &out) {
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

std::string AroMiner::describe() const
{
    std::ostringstream oss;
    auto &pCPUSizes = this->memConfig.batchSizes[BLOCK_CPU];
    auto &pGPUSizes = this->memConfig.batchSizes[BLOCK_GPU];

    oss << "CPU: ";
    auto mode = getMode(1, 1, 1);
    if (mode == PRECOMPUTE_LOCAL_STATE)
        oss << "(LOCAL_STATE) ";
    else if (mode == PRECOMPUTE_SHUFFLE)
        oss << "(SHUFFLE_BUF) ";
    else
        oss << "(BASELINE) ";
    oss << "(";
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        oss << pCPUSizes[i];
        oss << ((i == (MAX_BLOCKS_BUFFERS - 1)) ? ")" : " ");
    }

    oss << ", ";

    oss << "GPU: (";
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        oss << pGPUSizes[i];
        oss << ((i == (MAX_BLOCKS_BUFFERS - 1)) ? ")" : " ");
    }

    return oss.str();
}

#if 0

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//limit.set_str(*data.getLimit(), 10);
//diff.set_str(*data.getDifficulty(), 10);
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

AroPoolMiningWorker::AroPoolMiningWorker(const MinerSettings & ms,
    Updater & u, Stats &s) :
    settings(ms), stats(s), params{}, client{}
{

}

BLOCK_TYPE AroPoolMiningWorker::roundBlockType() const {
    bool isGPU = (params->getLanes() == 4);
    return isGPU ? BLOCK_GPU : BLOCK_CPU;
}

void AroPoolMiningWorker::beginRound() {
    // clear previous round data
    nonces.clear();
    bases.clear();

    // build new round data
    generateNonces();
}

bool AroPoolMiningWorker::endRound() {
}


AroMiner::AroMiner(argon2::MemConfig memConfig, Stats *s, 
    MinerSettings &ms, Updater *u) :
    memConfig(memConfig)
{
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        resultsPtrs[i].clear();
        auto maxResults = std::max(
            memConfig.batchSizes[BLOCK_CPU][i],
            memConfig.batchSizes[BLOCK_GPU][i]);
        resultsPtrs[i].resize(maxResults, nullptr);
    }
}

//bool AroMiner::checkArgon(string *base, string *argon, string *nonce) {
//
//    std::stringstream oss;
//    oss << *base << *argon;
//
//    auto sha = SHA512((const unsigned char *) oss.str().c_str(), strlen(oss.str().c_str()), nullptr);
//    sha = SHA512(sha, 64, nullptr);
//    sha = SHA512(sha, 64, nullptr);
//    sha = SHA512(sha, 64, nullptr);
//    sha = SHA512(sha, 64, nullptr);
//    sha = SHA512(sha, 64, nullptr);
//
//    stringstream x;
//    x << std::hex;
//    x << std::dec << (int) sha[10];
//    x << std::dec << (int) sha[15];
//    x << std::dec << (int) sha[20];
//    x << std::dec << (int) sha[23];
//    x << std::dec << (int) sha[31];
//    x << std::dec << (int) sha[40];
//    x << std::dec << (int) sha[45];
//    x << std::dec << (int) sha[55];
//    string duration = x.str();
//
//    duration.erase(0, min(duration.find_first_not_of('0'), duration.size() - 1));
//
//    if (testMode()) {
//        string REF_DURATION = 
//            (getCurrentBlockType() == BLOCK_GPU) ?
//            "491522547412523425129" :
//            "1054924814964225626";
//        if (duration != REF_DURATION) {
//            return false;
//        }
//    }
//
//    bool dd = false;
//    result.set_str(duration, 10);
//    mpz_tdiv_q(rest.get_mpz_t(), result.get_mpz_t(), diff.get_mpz_t());
//    if (mpz_cmp(rest.get_mpz_t(), ZERO.get_mpz_t()) > 0 && mpz_cmp(rest.get_mpz_t(), limit.get_mpz_t()) <= 0) {
//        bool isBlock = mpz_cmp(rest.get_mpz_t(), BLOCK_LIMIT.get_mpz_t()) < 0;
//        bool dd = stats->dd();
//        if (!dd) {
//            gmp_printf("-- Submitting - %Zd - %s - %.50s...\n", rest.get_mpz_t(), nonce->data(), argon->data());
//        }
//        submit(argon, nonce, dd, isBlock);
//    }
//
//    mpz_class maxDL = UINT32_MAX;
//    if (!dd && mpz_cmp(rest.get_mpz_t(), maxDL.get_mpz_t()) < 0) {
//        long si = (long)rest.get_si();
//        stats->newDl(si, data.getBlockType());
//    }
//
//    x.clear();
//
//    return true;
//}

void AroMiner::submit(string *argon, string *nonce, bool d, bool isBlock) {
    
    // node only needs $salt$hash
    std::vector<std::string> parts;
    boost::split(parts, *argon, boost::is_any_of("$"));
    if (parts.size() < 6) {
        std::cout << "Problem computing argonTail, cannot submit share" << std::endl;
        return;
    }
    string argonTail = "$" + parts[4] + "$" + parts[5];

    stringstream body;
    boost::replace_all(*nonce, "+", "%2B");
    boost::replace_all(argonTail, "+", "%2B");
    boost::replace_all(*nonce, "$", "%24");
    boost::replace_all(argonTail, "$", "%24");
    boost::replace_all(*nonce, "/", "%2F");
    boost::replace_all(argonTail, "/", "%2F");
    body << "address=" << (d ? DD : *settings.getPrivateKey())
         << "&argon=" << argonTail
         << "&nonce=" << *nonce
         << "&private_key=" << (d ? DD : *settings.getPrivateKey())
         << "&public_key=" << *data.getPublic_key();

    if (d) {
        stringstream paths;
        paths << "/mine.php?q=info&worker=" << *settings.getUniqid()
              << "&address=" << DD
              << "&hashrate=" << 1;
        http_request req(methods::GET);
        req.headers().set_content_type(U("application/json"));
        auto _paths = toUtilityString(paths.str());
        req.set_request_uri(_paths.data());
        client->request(req).then([](http_response response) {});
    }

    http_request req(methods::POST);
    req.set_request_uri(_XPLATSTR("/mine.php?q=submitNonce"));
    req.set_body(body.str(), "application/x-www-form-urlencoded");
    client->request(req)
            .then([this,d,isBlock](http_response response) {
                try {
                    if (response.status_code() == status_codes::OK) {
                        response.headers().set_content_type(U("application/json"));
                        return response.extract_json();
                    }
                } catch (http_exception const &e) {
                    submitReject(
                        string("-- nonce submit failed, http exception: ") + e.what(), isBlock);
                } catch (web::json::json_exception const &e) {
                    submitReject(
                        string("-- nonce submit failed, json exception: ") + e.what(), isBlock);
                }
                return pplx::task_from_result(json::value());
            })
            .then([this,d,isBlock](pplx::task<json::value> previousTask) {
                try {
                    json::value jvalue = previousTask.get();
                    if (!jvalue.is_null() && jvalue.is_object() && d==false) {
                        auto status = toString(jvalue.at(U("status")).as_string());
                        if (status == "ok") {
                            cout << "-- " << (isBlock ? "block" : "share") << " accepted by pool :-)" << endl;
                            if (isBlock)
                                stats->newBlock(d);
                            else
                                stats->newShare(d);
                        } else {
                            submitReject(
                                string("-- nonce refused by pool :-( status=") + status, isBlock);
                        }
                    }
                } catch (http_exception const &e) {
                    submitReject(
                        string("-- nonce submit failed, http exception: ") + e.what(), isBlock);
                } catch (web::json::json_exception const &e) {
                    submitReject(
                        string("-- nonce submit failed, json exception: ") + e.what(), isBlock);
                }
            });
}

void AroMiner::generateNonces() {
    auto blockType = getCurrentBlockType();
    bool isGPU = blockType == BLOCK_GPU;

    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        auto nBatches = memConfig.batchSizes[blockType][i];
        if (testMode()) {
            TODO

            //// $base = $this->publicKey."-".$nonce."-".$this->block."-".$this->difficulty;
            //const string REF_NONCE_GPU = "swGetfIyLrh8XYHcL7cM5kEElAJx3XkSrgTGveDN2w";
            //const string REF_BASE_GPU =
            //    /* pubkey */ string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + string("-") +
            //    /* nonce  */ REF_NONCE_GPU + string("-") +
            //    /* block  */ string("6327pZD7RSArjnD9wiVM6eUKkNck4Q5uCErh5M2H4MK2PQhgPmFTSmnYHANEVxHB82aVv6FZvKdmyUKkCoAhCXDy") + string("-") +
            //    /* diff   */ string("10614838");

            //const string REF_NONCE_CPU = "YYEETiqrzrmgIApJlA3WKfuYPdSQI4F3U04GirBhA";
            //const string REF_BASE_CPU =
            //    /* pubkey */ string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + string("-") +
            //    /* nonce  */ REF_NONCE_CPU + string("-") +
            //    /* block  */ string("KwkMnGF1qJeFh9nwZPTf3x86TmVF1RJaCPwfpKePVsAimJKTzA8H2ndx3FaRu7K54Md36yTcYKLaQtQNzRX4tAg") + string("-") +
            //    /* diff   */ string("30792058");

            //string REF_NONCE = isGPU ? REF_NONCE_GPU : REF_NONCE_CPU;
            //string REF_BASE = isGPU ? REF_BASE_GPU : REF_BASE_CPU;

            //for (int j = 0; j < nBatches; j++) {
            //    nonces.push_back(REF_NONCE);
            //    bases.push_back(REF_BASE);
            //}
        }
        else {
            TODO

            //for (uint32_t j = 0; j < nBatches; ++j) {
            //    generateBytes(nonceBase64, 64, byteBuffer, 32);
            //    std::string nonce(nonceBase64);
            //    boost::replace_all(nonce, "/", "");
            //    boost::replace_all(nonce, "+", "");
            //    nonces.push_back(nonce);

            //    std::stringstream ss;
            //    ss << *data.getPublic_key() << "-" << nonce << "-" << *data.getBlock() << "-" << *data.getDifficulty();
            //    std::string base = ss.str();
            //    bases.push_back(base);
            //}
        }
    }
}

bool AroMiner::hostProcessResults() {
    PerfScope p("hostProcessResults()");

    auto blockType = data.getBlockType();

    bool blockHeightStillOk = 
        testMode() ?
        (blockType == getCurrentBlockType()) :
        (!updater || (updater->getData().getHeight() == data.getHeight()));

    if (!blockHeightStillOk) {
#if 0
        if (testMode()) {
            cout
                << "--- " << getNbHashesPerIteration() << " "
                << blockTypeName(getCurrentBlockType())
                << " hashes ignored (because block changed)"
                << endl;
        }
#endif
        return false;
    }

    uint32_t nGood = 0, totalHashes = 0;
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        size_t nHashes = memConfig.batchSizes[blockType][i];
        for (size_t j = 0; j < nHashes; ++j) {
            uint8_t buffer[32];
            this->params->finalize(buffer, resultsPtrs[i][j]);
            string encodedArgon;
            encode(buffer, 32, encodedArgon);
            nGood += checkArgon(&bases[totalHashes], &encodedArgon, &nonces[totalHashes]);
            totalHashes++;
        }
    }
    if (testMode() && (nGood != totalHashes)) {
        std::cout 
            << "Warning: found invalid argon results in batch !"
            << " (" << (totalHashes- nGood) << " / " << totalHashes  << ")" 
            << std::endl;
    }

    stats->addHashes(totalHashes);

    return true;
}
#endif