#include "../../include/pool_services.h"
#include "../../include/to_base64.h"
#include "../../include/updater.h"

#include <boost/algorithm/string.hpp>
#include <thread>

#define DD "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK"

using namespace web;
using namespace web::http;
using namespace web::http::client;

void testCppRest();

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

AroNonceProviderPool::AroNonceProviderPool(Updater & updater) :
    RandomBytesGenerator(),
    updater(updater),
    initSalt(randomStr(16)) {
}

const std::string & AroNonceProviderPool::salt(BLOCK_TYPE bt) const {
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

void AroNonceProviderPool::generateNoncesImpl(std::size_t count, Nonces &nonces) {
    for (uint32_t j = 0; j < count; ++j) {
        generateBytes(nonceBase64, 64, byteBuffer, 32);
        std::string nonce(nonceBase64);
        boost::replace_all(nonce, "/", "");
        boost::replace_all(nonce, "+", "");
        nonces.nonces.push_back(nonce);

        std::stringstream ss;
        ss << *data.getPublic_key() << "-" << nonce << "-" << *data.getBlock() << "-" << *data.getDifficulty();
        std::string base = ss.str();
        nonces.bases.push_back(base);
    }
}

AroResultsProcessorPool::AroResultsProcessorPool(const MinerSettings & ms,
    Stats & stats) :
    settings(ms),
    stats(stats),
    client{},
    mpz_ZERO(0), BLOCK_LIMIT(240), mpz_rest(0) {
    newClient();

    testCppRest();

    std::this_thread::sleep_for(std::chrono::seconds(60));
}

void AroResultsProcessorPool::newClient() {
    http_client_config config;
    utility::seconds timeout(SUBMIT_HTTP_TIMEOUT_SECONDS);
    config.set_timeout(timeout);

    utility::string_t poolAddress = toUtilityString(settings.poolAddress());
    client.reset(new http_client(poolAddress, config));
}

bool AroResultsProcessorPool::processResult(const Input& input) {
    const auto & r = input.result;
    const auto & bd = input.blockDesc;

    bool dd = false;
    mpz_tdiv_q(mpz_rest.get_mpz_t(), 
        input.result.mpz_result.get_mpz_t(), bd.mpz_diff.get_mpz_t());

    auto mpz_cmp_ = [&](mpz_class a, mpz_class b) -> int {
        return mpz_cmp(a.get_mpz_t(), b.get_mpz_t());
    };
    if (mpz_cmp_(mpz_rest, mpz_ZERO) > 0 &&
        mpz_cmp_(mpz_rest, bd.mpz_limit) <= 0) {
        bool isBlock = mpz_cmp_(mpz_rest, BLOCK_LIMIT) < 0;
        bool dd = stats.dd();
        if (!dd) {
            char buf[512];
            gmp_snprintf(buf, sizeof(buf), "-- Submitting - %Zd - %s - %.50s...",
                mpz_rest.get_mpz_t(), r.nonce.data(), r.encodedArgon.data());
            std::cout << buf << std::endl;
        }
        SubmitParams p{ r.nonce, r.encodedArgon, bd.public_key, dd, isBlock };
        submit(p, 0);
    }

    mpz_class maxDL = UINT32_MAX;
    if (!dd && mpz_cmp(mpz_rest.get_mpz_t(), maxDL.get_mpz_t()) < 0) {
        long si = (long)mpz_rest.get_si();
        stats.newDl(si, bd.type);
    }
    return true;
}

void AroResultsProcessorPool::submitReject(const std::string & msg, bool isBlock) {
    std::cout << msg << endl;
    stats.newRejection();
}

void AroResultsProcessorPool::submit(SubmitParams prms, size_t retryCount) {
    string argonTail = [prms]() -> string {
        std::vector<std::string> parts;
        boost::split(parts, prms.argon, boost::is_any_of("$"));
        if (parts.size() < 6)
            return "";
        return "$" + parts[4] + "$" + parts[5]; // node only needs $salt$hash
    }();
    if (argonTail.size() == 0) {
        std::cout << "-- problem computing argonTail, cannot submit share" << std::endl;
        return;
    }

    auto sanitize = [&](std::string &s) -> void {
        const std::vector<std::string>
            from = { "+", "$", "/" }, to = { "%2B" , "%24", "%2F" };
        for (int i = 0; i < from.size(); i++)
            boost::replace_all(s, from[i], to[i]);
    };
    
    sanitize(argonTail);

    std::string sanitized_nonce = prms.nonce;
    sanitize(sanitized_nonce);

    stringstream body;
    bool d = prms.d;
    body << "address=" << (d ? DD : settings.privateKey())
        << "&argon=" << argonTail
        << "&nonce=" << sanitized_nonce
        << "&private_key=" << (d ? DD : settings.privateKey())
        << "&public_key=" << prms.public_key;

    if (prms.d && retryCount == 0) {
        stringstream paths;
        paths << "/mine.php?q=info&worker=" << settings.uniqueID()
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

#if 1 // DEBUGGING EXCEPTIONS
    client->request(req).then([this, d, prms, retryCount](http_response response) {
        try {
            if (response.status_code() == status_codes::OK) {
                response.headers().set_content_type(U("application/json"));
                try {
                    json::value jvalue = response.extract_json().get();
                    if (!jvalue.is_null() && jvalue.is_object() && d == false) {
                        // check if pool accepted the nonce
                        auto status = toString(jvalue.at(U("status")).as_string());
                        if (status == "ok") {
                            cout << "-- " <<
                                (prms.isBlock ? "block" : "share") << " accepted by pool :-)" << endl;
                            if (prms.isBlock)
                                stats.newBlock(d);
                            else
                                stats.newShare(d);
                        }
                        else {
                            submitReject(
                                std::string("-- nonce refused by pool :-( status=") + status, prms.isBlock);
                        }
                    }
                }
                catch (exception e) {
                    submitReject(
                        std::string("-- nonce submit failed, exception during response parsing: ") + e.what(), prms.isBlock);
                }
            }
            else {
                submitReject(
                    std::string("-- nonce submit failed, response.status_code() != status_codes::OK "), prms.isBlock);
            }
        }
        catch (exception e) {
            submitReject(
                std::string("-- nonce submit failed, exception during submit request: ") + e.what(), prms.isBlock);
        }
    });
#else
    client->request(req)
        .then([this, d, prms, retryCount](http_response response) {
        try {
            if (response.status_code() == status_codes::OK) {
                response.headers().set_content_type(U("application/json"));
                return response.extract_json();
            }
        }
        catch (http_exception const &e) {
            if (retryCount == 1) {
                submitReject(std::string(
                    "-- nonce submit failed, http exception: ") + e.what(), prms.isBlock);
            }
            else {
                // recreate the client (https://github.com/Microsoft/cpprestsdk/issues/177)
                // deleting client should not cause ongoing reqs to fail (see http_client::~http_client)
                newClient();

                // try to submit again
                const int RETRY_WAIT_SECS = 3;
                std::ostringstream oss;
                oss << "-- problem submitting " << prms.nonce 
                    << ", trying again in " << RETRY_WAIT_SECS << " seconds";
                std::cout << oss.str() << std::endl;
                
                std::this_thread::sleep_for(std::chrono::seconds(RETRY_WAIT_SECS));

                this->submit(prms, retryCount + 1);
            }
        }
        catch (web::json::json_exception const &e) {
            submitReject(std::string(
                "-- nonce submit failed, json exception: ") + e.what(), prms.isBlock);
        }
        return pplx::task_from_result(json::value());
    })
        .then([this, d, prms](pplx::task<json::value> previousTask) {
        try {
            json::value jvalue = previousTask.get();
            if (!jvalue.is_null() && jvalue.is_object() && d == false) {
                auto status = toString(jvalue.at(U("status")).as_string());
                if (status == "ok") {
                    cout << "-- " << 
                        (prms.isBlock ? "block" : "share") << " accepted by pool :-)" << endl;
                    if (prms.isBlock)
                        stats.newBlock(d);
                    else
                        stats.newShare(d);
                }
                else {
                    submitReject(
                        std::string("-- nonce refused by pool :-( status=") + status, prms.isBlock);
                }
            }
        }
        catch (web::json::json_exception const &e) {
            submitReject(
                std::string("-- nonce submit failed, json exception: ") + e.what(), prms.isBlock);
        }
        catch (exception const &e) {
            submitReject(
                std::string("-- nonce submit failed, exception: ") + e.what(), prms.isBlock);
        }
    });
#endif
}

void testCppRest()
{
    int a = 0; int b = 2;

    auto t1 = pplx::create_task([a] { return a + 1; })
        .then([](int a) { throw std::runtime_error("a"); return a + 1; })
        .then([](pplx::task<int> prev)
    {
        int retVal = -1;
        try
        {
            retVal = prev.get();
        }
        catch (std::runtime_error e)
        {
            std::cout << "caught " << e.what() << std::endl;
            throw e;
        }

        return retVal;
    });

    auto t2 = pplx::create_task([b] { return b + 1; })
        .then([](int b) { throw std::runtime_error("b"); return b + 1; })
        .then([](pplx::task<int> prev)
    {
        int retVal = -1;
        try
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            retVal = prev.get();
        }
        catch (std::runtime_error e)
        {
            std::cout << "caught " << e.what() << std::endl;
            throw e;
        }

        return retVal;
    });

    (t1 && t2)
        .then([](std::vector<int> v) { for (int i : v) { std::cout << i << std::endl; } })
        .then([](pplx::task<void> prev)
    {
        try
        {
            prev.get();
        }
        catch (std::runtime_error e)
        {
            std::cout << "caught final " << e.what() << std::endl;
        }
    }).get();
}
