//
// Created by guli on 01/02/18.
//
#define DD "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK"

#include <boost/algorithm/string.hpp>

#include <iomanip>
#include <openssl/sha.h>
#include <thread>

#include "../../include/miner.h"
#include "../../include/timer.h"

using namespace std;

static int b64_byte_to_char(unsigned x) {
    return (LT(x, 26) & (x + 'A')) |
           (GE(x, 26) & LT(x, 52) & (x + ('a' - 26))) |
           (GE(x, 52) & LT(x, 62) & (x + ('0' - 52))) | (EQ(x, 62) & '+') |
           (EQ(x, 63) & '/');
}

// to fix mixed up messages at start
bool gMiningStarted = false;

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

#ifdef TEST_GPU_BLOCK
// GPU block test
const string REF_NONCE = "swGetfIyLrh8XYHcL7cM5kEElAJx3XkSrgTGveDN2w";
//$base = $this->publicKey."-".$nonce."-".$this->block."-".$this->difficulty;
const string REF_BASE =
    string("PZ8Tyr4Nx8MHsRAGMpZmZ6TWY63dXWSCy7AEg3h9oYjeR74yj73q3gPxbxq9R3nxSSUV4KKgu1sQZu9Qj9v2q2HhT5H3LTHwW7HzAA28SjWFdzkNoovBMncD") + string("-") + // publicKey
    REF_NONCE  + string("-") + // nonce
    string("6327pZD7RSArjnD9wiVM6eUKkNck4Q5uCErh5M2H4MK2PQhgPmFTSmnYHANEVxHB82aVv6FZvKdmyUKkCoAhCXDy") + string("-") + // block
    string("10614838");  // difficulty
#endif

void Miner::buildBatch() {
    for (int j = 0; j < *settings->getBatchSize(); ++j) {
#ifdef TEST_GPU_BLOCK
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

#ifdef TEST_GPU_BLOCK
    const string REF_DURATION = "491522547412523425129";

    if (duration != REF_DURATION) {
        static bool errShown = false;
        if (!errShown) {
            std::cout << std::endl << "-------------- TEST_GPU_BLOCK: invalid duration: " << duration << " / " << REF_DURATION << std::endl;
            std::cout << std::endl << "argon=" << *argon << std::endl;
            errShown = true;
            exit(1);
        }
        duration = "10000000000000000000";
    }
#endif

    bool submitted = false;
    result.set_str(duration, 10);
    mpz_tdiv_q(rest.get_mpz_t(), result.get_mpz_t(), diff.get_mpz_t());
    if (mpz_cmp(rest.get_mpz_t(), ZERO.get_mpz_t()) > 0 && mpz_cmp(rest.get_mpz_t(), limit.get_mpz_t()) <= 0) {
        submitted = true;
        bool d = mpz_cmp(rest.get_mpz_t(), BLOCK_LIMIT.get_mpz_t()) < 0 ? stats->newBlock() : stats->newShare();
        gmp_printf("Submitting - %Zd - %s - %s\n", rest.get_mpz_t(), nonce->data(), argon->data());
        submit(argon, nonce, d);
    }

    mpz_class maxLong = LONG_MAX;
    if (mpz_cmp(rest.get_mpz_t(), maxLong.get_mpz_t()) < 0) {
        long si = rest.get_si();
        stats->newDl(si);
    }


    x.clear();
}

void Miner::submit(string *argon, string *nonce, bool d) {
    
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
    http_request req(methods::POST);
    req.set_request_uri(_XPLATSTR("/mine.php?q=submitNonce"));
    req.set_body(body.str(), "application/x-www-form-urlencoded");
    client->request(req)
            .then([this](http_response response) {
                try {
                    if (response.status_code() == status_codes::OK) {
                        response.headers().set_content_type(L"application/json");
                        return response.extract_json();
                    }
                } catch (http_exception const &e) {
                    cout << e.what() << endl;
                } catch (web::json::json_exception const &e) {
                    cerr << e.what() << endl;
                }
                return pplx::task_from_result(json::value());
            })
            .then([this](pplx::task<json::value> previousTask) {
                try {
                    json::value jvalue = previousTask.get();
                    if (!jvalue.is_null() && jvalue.is_object()) {
                        auto status = toString(jvalue.at(L"status").as_string());
                        if (status == "ok") {
                            cout << "nonce accepted by pool !!!!!" << endl;
                        } else {
                            cout << "nonce refused by pool :(:(:(" << endl;
                            cout << status << endl;
                            stats->newRejection();
                        }
                    }
                } catch (http_exception const &e) {
                    cout << e.what() << endl;
                } catch (web::json::json_exception const &e) {
                    cerr << e.what() << endl;
                }
            })
            .wait();
}

char *Miner::encode(void *res, size_t reslen) {
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
    auto salt = new char[32];
    const char *saltRaw = (const char *)params->getSalt();
    to_base64(salt, 32, saltRaw, params->getSaltLength());
    ss << salt;
    auto hash = new char[512];
    to_base64(hash, 512, res, reslen);
    ss << "$";
    ss << hash;


    std::string str = ss.str();
    char *cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    free(salt);
    free(hash);
    return cstr;
}

void Miner::hostPrepareTaskData() {
    // see if block changed
    if (!data.isValid() || data.isNewBlock(updater->getData().getBlock())) {
        data = updater->getData();
        while (!data.isValid()) {
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
    size_t size = *settings->getBatchSize();
    uint8_t buffer[32];
    for (size_t j = 0; j < size; ++j) {
        this->params->finalize(buffer, resultBuffers[j]);
        char *openClEnc = encode(buffer, 32);
        string encodedArgon(openClEnc);
        argons.push_back(encodedArgon);
    }

    // now check each one (to see if we need to submit it or not)
    for (int j = 0; j < *settings->getBatchSize(); ++j) {
        checkArgon(&bases[j], &argons[j], &nonces[j]);
    }

    // update stats
    stats->addHashes((long)(*settings->getBatchSize()));
}