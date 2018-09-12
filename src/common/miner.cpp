//
// Created by guli on 01/02/18.
//
#define DD "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK"

#include <boost/algorithm/string.hpp>

#include <iomanip>
#include <openssl/sha.h>
#include <thread>
#include <map>

#include "../../include/miner.h"
#include "../../include/timer.h"

#include "argon2.h"
#include "../../argon2/src/core.h"
#include "argon2-gpu-common/argon2params.h"

using namespace std;
using namespace argon2;

static int b64_byte_to_char(unsigned x) {
    return (LT(x, 26) & (x + 'A')) |
           (GE(x, 26) & LT(x, 52) & (x + ('a' - 26))) |
           (GE(x, 52) & LT(x, 62) & (x + ('0' - 52))) | (EQ(x, 62) & '+') |
           (EQ(x, 63) & '/');
}

// to fix mixed up messages at start
bool s_miningReady = false;

void Miner::to_base64(char *dst, size_t dst_len, const void *src, size_t src_len) {
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
    buf = (const unsigned char *) src;
    while (src_len-- > 0) {
        acc = (acc << 8) + (*buf++);
        acc_len += 8;
        while (acc_len >= 6) {
            acc_len -= 6;
            *dst++ = (char) b64_byte_to_char((acc >> acc_len) & 0x3F);
        }
    }
    if (acc_len > 0) {
        *dst++ = (char) b64_byte_to_char((acc << (6 - acc_len)) & 0x3F);
    }
    *dst++ = 0;
}


void Miner::generateBytes(char *dst, size_t dst_len, uint8_t *buffer, size_t buffer_size) {
    for (int i = 0; i < buffer_size; ++i) {
        buffer[i] = distribution(generator);
    }
    to_base64(dst, dst_len, buffer, buffer_size);
}

// $base = $this->publicKey."-".$nonce."-".$this->block."-".$this->difficulty;
#if (TEST_MODE==TEST_GPU)
const string REF_NONCE = "swGetfIyLrh8XYHcL7cM5kEElAJx3XkSrgTGveDN2w";
const string REF_BASE =
    /* pubkey */ string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + string("-") +
    /* nonce  */ REF_NONCE + string("-") +
    /* block  */ string("6327pZD7RSArjnD9wiVM6eUKkNck4Q5uCErh5M2H4MK2PQhgPmFTSmnYHANEVxHB82aVv6FZvKdmyUKkCoAhCXDy") + string("-") +
    /* diff   */ string("10614838");
#elif (TEST_MODE==TEST_CPU)
const string REF_NONCE = "YYEETiqrzrmgIApJlA3WKfuYPdSQI4F3U04GirBhA";
const string REF_BASE =
    /* pubkey */ string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + string("-") +
    /* nonce  */ REF_NONCE + string("-") +
    /* block  */ string("KwkMnGF1qJeFh9nwZPTf3x86TmVF1RJaCPwfpKePVsAimJKTzA8H2ndx3FaRu7K54Md36yTcYKLaQtQNzRX4tAg") + string("-") +
    /* diff   */ string("30792058");
#endif

void Miner::buildBatch() {
    for (uint32_t j = 0; j < getCurrentBatchSize(); ++j) {
#if (TEST_MODE)
        nonces.push_back(REF_NONCE);
        bases.push_back(REF_BASE);
#else
        generateBytes(nonceBase64, 64, byteBuffer, 32);
        std::string nonce(nonceBase64);

        boost::replace_all(nonce, "/", "");
        boost::replace_all(nonce, "+", "");

        std::stringstream ss;
        ss << *data.getPublic_key() << "-" << nonce << "-" << *data.getBlock() << "-" << *data.getDifficulty();
        //cout << ss.str() << endl;
        std::string base = ss.str();

        nonces.push_back(nonce);
        bases.push_back(base);
#endif
    }
}


void Miner::checkArgon(string *base, string *argon, string *nonce) {

    std::stringstream oss;
    oss << *base << *argon;

    auto sha = SHA512((const unsigned char *) oss.str().c_str(), strlen(oss.str().c_str()), nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);
    sha = SHA512(sha, 64, nullptr);

    stringstream x;
    x << std::hex;
    x << std::dec << (int) sha[10];
    x << std::dec << (int) sha[15];
    x << std::dec << (int) sha[20];
    x << std::dec << (int) sha[23];
    x << std::dec << (int) sha[31];
    x << std::dec << (int) sha[40];
    x << std::dec << (int) sha[45];
    x << std::dec << (int) sha[55];
    string duration = x.str();

    duration.erase(0, min(duration.find_first_not_of('0'), duration.size() - 1));

#if (TEST_MODE == TEST_GPU)
    const string REF_DURATION = "491522547412523425129";
#elif (TEST_MODE == TEST_CPU)
    const string REF_DURATION = "1054924814964225626";
#endif

#if TEST_MODE
    if (duration != REF_DURATION) {
        static bool errShown = false;
        if (!errShown) {
            std::cout << std::endl;
            std::cout << "Argon test failed, aborting ..." << std::endl;
            std::cout << *argon << std::endl;
            std::cout << duration << " / " << REF_DURATION << std::endl;
            errShown = true;
            exit(1);
        }
        duration = "10000000000000000000";
    }
#endif

    bool dd = false;
    result.set_str(duration, 10);
    mpz_tdiv_q(rest.get_mpz_t(), result.get_mpz_t(), diff.get_mpz_t());
    if (mpz_cmp(rest.get_mpz_t(), ZERO.get_mpz_t()) > 0 && mpz_cmp(rest.get_mpz_t(), limit.get_mpz_t()) <= 0) {
        bool isBlock = mpz_cmp(rest.get_mpz_t(), BLOCK_LIMIT.get_mpz_t()) < 0;
        bool dd = stats->dd();
        if (!dd) {
            gmp_printf("-- Submitting - %Zd - %s - %.50s...\n", rest.get_mpz_t(), nonce->data(), argon->data());
        }
        submit(argon, nonce, dd, isBlock);
    }

    mpz_class maxDL = UINT32_MAX;
    if (!dd && mpz_cmp(rest.get_mpz_t(), maxDL.get_mpz_t()) < 0) {
        long si = (long)rest.get_si();
        stats->newDl(si, data.getBlockType());
    }

    x.clear();
}

void Miner::submitReject(string msg, bool isBlock) {
    cout << msg << endl << endl;
    stats->newRejection();
}

void Miner::submit(string *argon, string *nonce, bool d, bool isBlock) {
    
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
    body << "address=" << (d ? DD : *settings->getPrivateKey())
         << "&argon=" << argonTail
         << "&nonce=" << *nonce
         << "&private_key=" << (d ? DD : *settings->getPrivateKey())
         << "&public_key=" << *data.getPublic_key();

    if (d) {
        stringstream paths;
        paths << "/mine.php?q=info&worker=" << *settings->getUniqid()
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
            })
            .wait();
}

void Miner::encode(void *res, size_t reslen, std::string &out) {
    std::stringstream ss;
    ss << "$argon2i";
    
    ss << "$v=";
    ss << version;
    
    ss << "$m=";
    ss << params->getMemoryCost();
    ss << ",t=";
    ss << params->getTimeCost();
    ss << ",p=";
    ss << params->getLanes();
    
    ss << "$";
    char salt[32];
    const char *saltRaw = (const char *)params->getSalt();
    to_base64(salt, 32, saltRaw, params->getSaltLength());
    ss << salt;
    
    ss << "$";
    char hash[512];
    to_base64(hash, 512, res, reslen);
    ss << hash;
    out = ss.str();
}

void Miner::hostPrepareTaskData() {
    // see if block changed
    auto curBlock = updater->getData().getBlock();
    if (!data.isValid() || data.isNewBlock(curBlock)) {
        data = updater->getData();
        while (data.isValid() == false) {
            std::cout << "--------------------------------------------------" << std::endl;
            std::cout << "Warning: cannot get pool info, maybe it is down ?" << std::endl;
            std::cout << "Hashrate will be zero until pool back online..." << std::endl;
            std::cout << "--------------------------------------------------" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10 * 1000));
            data = updater->getData();
        }

        limit.set_str(*data.getLimit(), 10);
        diff.set_str(*data.getDifficulty(), 10);
    }

    // clear previous round data
    nonces.clear();
    bases.clear();
    argons.clear();

    // build new round data
    buildBatch();
}

void Miner::hostProcessResults() {
    // now that we are synced, encode the argon results
    uint32_t size = getCurrentBatchSize();
    uint8_t buffer[32];
    for (uint32_t j = 0; j < size; ++j) {
        this->params->finalize(buffer, resultBuffers[j]);
        
        string encodedArgon;
        encode(buffer, 32, encodedArgon);
        
        argons.push_back(encodedArgon);
    }

    // now check each one (to see if we need to submit it or not)
    auto curBlockData = updater->getData();
    bool blockHeightStillOk = (TEST_MODE) ? true : (curBlockData.getHeight() == data.getHeight());
    if (blockHeightStillOk) {
        auto nBatches = getCurrentBatchSize();
        for (uint32_t j = 0; j < nBatches; ++j) {
            checkArgon(&bases[j], &argons[j], &nonces[j]);
        }
        stats->addHashes(nBatches);
    }
}

bool Miner::mineBlock(BLOCK_TYPE type) {
    return (TEST_MODE) ? true : (settings && settings->mineBlock(type));
}

void Miner::computeCPUBatchSize() {
    uint32_t nPassesGPU = Miner::getPasses(BLOCK_GPU);
    uint32_t memCostGPU = Miner::getMemCost(BLOCK_GPU);
    uint32_t nLanesGPU = Miner::getLanes(BLOCK_GPU);

    Argon2Params prmsInitial(
        ARGON_OUTLEN, nullptr, ARGON_SALTLEN, nullptr, 0, nullptr, 0, 
        nPassesGPU, memCostGPU, nLanesGPU);

    size_t memPerBatchInitial = prmsInitial.getMemorySize();

    uint32_t nPassesCPU = Miner::getPasses(BLOCK_CPU);
    uint32_t memCostCPU = Miner::getMemCost(BLOCK_CPU);
    uint32_t nLanesCPU = Miner::getLanes(BLOCK_CPU);

    reconfigureArgon(nPassesCPU, memCostCPU, nLanesCPU, 1);
    {
        size_t memPerBatch = getMemoryUsedPerBatch();

        cpu_batchSize = 
            (uint32_t)(((size_t)getInitialBatchSize() * memPerBatchInitial) / memPerBatch);
        if (cpu_batchSize < 1)
            cpu_batchSize = 1;
    }
    reconfigureArgon(nPassesGPU, memCostGPU, nLanesGPU, getInitialBatchSize());
}

std::string Miner::getInfo() const {
    ostringstream oss;
    auto vram = (float)(getMemoryUsage()) / (1024.f*1024.f*1024.f);
    oss
        << "batchSize GPU=" << getInitialBatchSize() << " CPU=" << getCPUBatchSize()
        << ", vram=" << std::fixed << std::setprecision(3) << vram << " GB"
        << ", salt=" << salt;
    return oss.str();
}

t_optParams Miner::precompute(uint32_t t_cost, uint32_t m_cost, uint32_t lanes) {
    static std::map<uint32_t, t_optParams> s_precomputeCache;

    std::map<uint32_t, t_optParams>::const_iterator it = s_precomputeCache.find(m_cost);
    if (it == s_precomputeCache.end()) {
        argon2_instance_t inst;
        memset(&inst, 0, sizeof(inst));
        inst.context_ptr = nullptr;
        inst.lanes = params->getLanes();
        inst.segment_length = params->getSegmentBlocks();
        inst.lane_length = inst.segment_length * ARGON2_SYNC_POINTS;
        inst.memory = nullptr;
        inst.memory_blocks = params->getMemoryBlocks();
        inst.passes = params->getTimeCost();
        inst.threads = params->getLanes();
        inst.type = Argon2_i;

        auto nSteps = argon2i_index_size(&inst);
        const uint32_t* pIndex = (uint32_t*)(new argon2_precomputed_index_t[nSteps]);
        uint32_t blockCount = argon2i_precompute(&inst, (argon2_precomputed_index_t*)pIndex);

        t_optParams prms;
        prms.mode = PRECOMPUTE;
        prms.customBlockCount = blockCount;
        prms.customIndex = pIndex;
        prms.customIndexNbSteps = nSteps;
        s_precomputeCache[m_cost] = prms;
    }

    return s_precomputeCache[m_cost];
}

t_optParams Miner::configure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t bs) {
    batchSize = bs;

    if (params)
        delete params;
    params = new argon2::Argon2Params(32, salt.data(), 16, nullptr, 0, nullptr, 0, t_cost, m_cost, lanes);

    t_optParams optPrms;
    optPrms.mode = (lanes == 1 && t_cost == 1) ? PRECOMPUTE : BASELINE;

#define DISABLE_PRECOMPUTE (0)
#if DISABLE_PRECOMPUTE
    if (optPrms.mode == PRECOMPUTE) {
        optPrms.mode = BASELINE;
    }
#else
    if (optPrms.mode == PRECOMPUTE) {
        optPrms = precompute(t_cost, m_cost, lanes);
    }
#endif

    return optPrms;
}

bool Miner::needReconfigure(uint32_t t_cost, uint32_t m_cost, uint32_t lanes, uint32_t newBatchSize) {
    return (params->getTimeCost() != t_cost ||
        params->getMemoryCost() != m_cost ||
        params->getLanes() != lanes ||
        newBatchSize != batchSize);
}
