#include "../../include/pool_services.h"
#include "../../include/to_base64.h"
#include "../../include/updater.h"

#include <boost/algorithm/string.hpp>
#include <thread>
#include <sstream>

#define DD "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK"

using namespace web;
using namespace web::http;
using namespace web::http::client;

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
    BLOCK_LIMIT(240), mpz_ZERO(0), mpz_rest(0) {
    newClient();
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
    auto sanitize = [&](std::string &s) -> void {
        const std::vector<std::string>
            from = { "+", "$", "/" }, to = { "%2B" , "%24", "%2F" };
        for (int i = 0; i < from.size(); i++)
            boost::replace_all(s, from[i], to[i]);
    };

    string argonTail = [prms]() -> string {
        std::vector<std::string> parts;
        boost::split(parts, prms.argon, boost::is_any_of("$"));
        if (parts.size() < 6)
            return "";
        return "$" + parts[4] + "$" + parts[5]; // node only needs $salt$hash
    }();
    sanitize(argonTail);
    if (argonTail.size() == 0) {
        std::cout << "-- problem computing argonTail, cannot submit share" << std::endl;
        return;
    }
    
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
        client->request(req).then([](pplx::task<http_response> response) {
            try { response.get(); }
            catch (const std::exception &) {};
        });
    }

    http_request req(methods::POST);
    req.set_request_uri(_XPLATSTR("/mine.php?q=submitNonce"));
    req.set_body(body.str(), "application/x-www-form-urlencoded");

    client->request(req)
    .then([](http_response response) 
    {
        if (response.status_code() == status_codes::OK) {
            response.headers().set_content_type(U("application/json"));
            return response.extract_json();
        }
        return pplx::task_from_result(json::value());
    })
    .then([this, prms, d, retryCount](pplx::task<json::value> previousTask) -> void
    {
        auto retry = [this, d, prms, retryCount]() -> void {
            // recreate the client (https://github.com/Microsoft/cpprestsdk/issues/177)
            // deleting client should not cause ongoing reqs to fail (see http_client::~http_client)
            newClient();

            // wait
            const int RETRY_WAIT_SECS = 3;
            if (!d) {
                std::ostringstream oss;
                oss << "-- problem submitting " << prms.nonce
                    << ", trying again in " << RETRY_WAIT_SECS << " seconds";
                std::cout << oss.str() << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::seconds(RETRY_WAIT_SECS));

            // submit again
            this->submit(prms, retryCount + 1);
        };

        auto handleException = [this, d, prms](const std::string & msg, const std::exception & e) -> void {
            std::ostringstream oss;
            if (!d) {
                oss << "-- nonce submit failed, " << msg << " " << e.what() << std::endl;
                submitReject(oss.str(), prms.isBlock);
            }
        };

        try {
            const json::value &jvalue = previousTask.get();
            bool jsonOk = !jvalue.is_null() && jvalue.is_object();
            if (!jsonOk) {
                if (retryCount == 0)
                    retry();
                else
                    throw std::logic_error("pool returned empty JSON");
                return;
            }
            
            if (d)
                return;

            auto status = toString(jvalue.at(U("status")).as_string());
            if (status == "ok") {
                cout << "-- " <<
                    (prms.isBlock ? "block" : "share") << " accepted by pool :-)" << endl;
                prms.isBlock ? stats.newBlock(d) : stats.newShare(d);
            }
            else {
                submitReject(
                    std::string("-- nonce refused by pool :-( status=") + status, prms.isBlock);
            }
        }
        catch (const web::http::http_exception & e) {
            if (retryCount == 0)
                retry();
            else
                handleException("http_exception", e);
        }
        catch (const web::json::json_exception & e) {
            handleException("json_exception", e);
        }
        catch (const std::exception & e) {
            handleException("exception", e);
        }
    });
}
