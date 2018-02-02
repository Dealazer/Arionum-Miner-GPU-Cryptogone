#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

#include "argon2-gpu-common/argon2params.h"
#include "argon2-opencl/processingunit.h"
#include "argon2-cuda/processingunit.h"
#include "argon2-cuda/cudaexception.h"
#include <openssl/sha.h>
#include <iomanip>
#include <gmp.h>
#include <gmpxx.h>
#include <time.h>
#include <thread>

#include <cpprest/http_client.h>
#include <cpprest/json.h>

#pragma comment(lib, "cpprest110_1_1")

using namespace web;
using namespace web::http;
using namespace web::http::client;


extern "C" {
#include "argon2-opencl/myopencl.h"
}

#include "argon2.h"

#include <iostream>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <boost/algorithm/string/replace.hpp>
#include <chrono>

#define EQ(x, y) ((((0U - ((unsigned)(x) ^ (unsigned)(y))) >> 8) & 0xFF) ^ 0xFF)
#define GT(x, y) ((((unsigned)(y) - (unsigned)(x)) >> 8) & 0xFF)
#define GE(x, y) (GT(y, x) ^ 0xFF)
#define LT(x, y) GT(y, x)
#define LE(x, y) GE(y, x)

using namespace argon2;
using namespace libcommandline;
using namespace std;

struct UpdateData {
    string status;
    string difficulty;
    signed long int limit = 0;
    string block;
    string public_key;
} updateData;

struct MinerData {
    string private_key = "4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL";
    unsigned long hashes = 0;
    unsigned long duration = 0;
    unsigned long totalHashes = 0;
    unsigned long shares = 0;
    unsigned long blocks = 0;
    unsigned long batchDuration = 0;
    unsigned long batchCount = 0;
    unsigned long rejected = 0;
} minerData;

static mpz_t ZERO;
static mpz_t BLOCK_LIMIT;
static string uniqid;

static void parseUpdateJson(const json::value &jvalue) {
    if (!jvalue.is_null() && jvalue.is_object()) {
        string status = jvalue.at(U("status")).as_string();
        if (strcmp(status.data(), "ok") == 0) {
            json::value data = jvalue.at(U("data"));
            if (data.is_object()) {
                string difficulty = data.at(U("difficulty")).as_string();
                string block = data.at(U("block")).as_string();
                string public_key = data.at(U("public_key")).as_string();
                int limit = data.at(U("limit")).as_integer();


                if (std::strcmp(updateData.block.data(), block.data()) != 0) {
                    std::cout << "NEW BLOCK FOUND" << std::endl;
                    std::cout << "status =>" << status << std::endl;
                    std::cout << "difficulty =>" << difficulty << std::endl;
                    std::cout << "block =>" << block << std::endl;
                    std::cout << "limit =>" << limit << std::endl;
                    std::cout << "public_key =>" << public_key << std::endl;
                    updateData.block = block;
                    updateData.difficulty = difficulty;
                    updateData.limit = limit;
                    updateData.public_key = public_key;
                }
            }
        } else {
            cerr << "Unable to get result" << endl;
        }
    }
}


struct Arguments {
    bool showHelp = false;
    bool listDevices = false;
    std::size_t deviceIndex = 0;
    int batchSize = 200;
};

static CommandLineParser<Arguments> buildCmdLineParser() {
    static const auto positional = PositionalArgumentHandler<Arguments>(
            [](Arguments &, const std::string &) {});

    std::vector<const CommandLineOption<Arguments> *> options{
            new FlagOption<Arguments>(
                    [](Arguments &state) { state.listDevices = true; },
                    "list-devices", 'l', "list all available devices and exit"),

            new ArgumentOption<Arguments>(
                    makeNumericHandler<Arguments, std::size_t>([](Arguments &state, std::size_t index) {
                        state.deviceIndex = (std::size_t) index;
                    }), "device", 'd', "use device with index INDEX", "0", "INDEX"),

            new ArgumentOption<Arguments>(
                    makeNumericHandler<Arguments, int>([](Arguments &state, int index) {
                        state.batchSize = (int) index;
                    }), "batchSize", 'b', "batch size", "200", "SIZE"),

            new FlagOption<Arguments>(
                    [](Arguments &state) { state.showHelp = true; },
                    "help", '?', "show this help and exit")
    };

    return CommandLineParser<Arguments>(
            "A tool for testing the argon2-opencl and argon2-cuda libraries.",
            positional, options);
}

static int b64_byte_to_char(unsigned x) {
    return (LT(x, 26) & (x + 'A')) |
           (GE(x, 26) & LT(x, 52) & (x + ('a' - 26))) |
           (GE(x, 52) & LT(x, 62) & (x + ('0' - 52))) | (EQ(x, 62) & '+') |
           (EQ(x, 63) & '/');
}

/*
 * Convert some bytes to Base64. 'dst_len' is the length (in characters)
 * of the output buffer 'dst'; if that buffer is not large enough to
 * receive the result (including the terminating 0), then (size_t)-1
 * is returned. Otherwise, the zero-terminated Base64 string is written
 * in the buffer, and the output length (counted WITHOUT the terminating
 * zero) is returned.
 */
static size_t to_base64(char *dst, size_t dst_len, const void *src,
                        size_t src_len) {
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
        return (size_t) -1;
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
    return olen;
}

static void generateBytes(char *dst, size_t dst_len) {
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<uint8_t> distribution(0, 255);
    std::vector<uint8_t> bytes;
    for (int i = 0; i < 32; ++i) {
        bytes.push_back(distribution(generator));
    }
    to_base64(dst, dst_len, bytes.data(), 32);
}

static void buildBatch(std::vector<std::string> *nonces,
                       std::vector<std::string> *bases,
                       int batchSize) {
    for (int j = 0; j < batchSize; ++j) {
        auto nonceBase64 = new char[64];
        generateBytes(nonceBase64, 64);
        std::string nonce(nonceBase64);
        boost::replace_all(nonce, "/", "");
        boost::replace_all(nonce, "+", "");

        std::stringstream ss;
        ss << updateData.public_key << "-" << nonce << "-" << updateData.block << "-" << updateData.difficulty;
        //cout << ss.str() << endl;
        std::string base = ss.str();

        nonces->push_back(nonce);
        bases->push_back(base);
        free(nonceBase64);
    }

}

static char *encode(argon2::opencl::ProgramContext *progCtx, argon2::Argon2Params *params, void *res, size_t reslen) {
    std::stringstream ss;
    ss << "$argon2i";
    ss << "$v=";
    ss << progCtx->getArgon2Version();
    ss << "$m=";
    ss << params->getMemoryCost();
    ss << ",t=";
    ss << params->getTimeCost();
    ss << ",p=";
    ss << params->getLanes();
    ss << "$";
    auto salt = new char[32];
    to_base64(salt, 32, params->getSalt(), params->getSaltLength());
    ss << salt;
    auto hash = new char[64];
    to_base64(hash, 64, res, reslen);
    ss << "$";
    ss << hash;


    std::string str = ss.str();
    char *cstr = new char[str.length() + 1];
    strcpy(cstr, str.c_str());
    free(salt);
    free(hash);
    return cstr;
}

static const char alphanum[] =
        "0123456789"
                "!@#$%^&*"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz";

static int stringLength = sizeof(alphanum) - 1;

static char genRandom(int v) {
    return alphanum[v];
}

static std::string randomStr(int length) {
    std::stringstream ss;
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(25, 63); // define the range

    for (int i = 0; i < length; ++i) {
        ss << genRandom(distr(eng) % stringLength);
    }
    return ss.str();
}

static void computeHash(argon2::opencl::ProcessingUnit *unit,
                        argon2::opencl::ProgramContext *progCtx,
                        argon2::Argon2Params *params,
                        std::vector<std::string> *bases,
                        std::vector<std::string> *argons,
                        int batchSize) {


    for (int j = 0; j < batchSize; ++j) {
        std::string data = bases->at(j);
        unit->setPassword(j, data.data(), data.length());
    }
    unit->beginProcessing();
    unit->endProcessing();
    auto buffer = std::unique_ptr<std::uint8_t[]>(new std::uint8_t[32]);
    for (int j = 0; j < batchSize; ++j) {
        unit->getHash(j, buffer.get());
        char *openClEnc = encode(progCtx, params, buffer.get(), 32);
        argons->push_back(std::string(openClEnc));
    }
}

static void updateInfoRequest(http_client &client) {
    stringstream paths;
    double rate = ((minerData.hashes * 1000.0) / minerData.duration);
    paths << "/mine.php?q=info&worker=" << uniqid << "&address=" << minerData.private_key << "&hashrate="
          << rate;
    //cout << "PATH="<< paths.str() << endl;
    http_request req(methods::GET);
    req.headers().set_content_type("application/json");
    req.set_request_uri(U(paths.str().data()));
    client.request(req)
            .then([](http_response response) {
                if (response.status_code() == status_codes::OK) {
                    response.headers().set_content_type("application/json");
                    return response.extract_json();
                }
                return pplx::task_from_result(json::value());
            })
            .then([](pplx::task<json::value> previousTask) {
                try {
                    parseUpdateJson(previousTask.get());
                }
                catch (http_exception const &e) {
                    cout << e.what() << endl;
                }
            })
            .wait();
}

static void submit(string *argon, string *nonce) {
    http_client client(U("http://aropool.com"));
    //http_client client(U("http://localhost:8888"));
    //cout << *nonce << endl;
    //cout << *argon << endl;
    stringstream body;
    boost::replace_all(*nonce, "+", "%2B");
    boost::replace_all(*argon, "+", "%2B");
    boost::replace_all(*nonce, "$", "%24");
    boost::replace_all(*argon, "$", "%24");
    boost::replace_all(*nonce, "/", "%2F");
    boost::replace_all(*argon, "/", "%2F");
    body << "address=" << minerData.private_key.data() << "&argon=" << *argon << "&nonce=" << *nonce << "&private_key="
         << minerData.private_key << "&public_key=" << updateData.public_key;

    cout << body.str() << endl;

    http_request req(methods::POST);
    req.set_request_uri(U("/mine.php?q=submitNonce"));
    req.set_body(body.str(), "application/x-www-form-urlencoded");
    client.request(req)
            .then([](http_response response) {
                if (response.status_code() == status_codes::OK) {
                    response.headers().set_content_type("application/json");
                    return response.extract_json();
                }
                return pplx::task_from_result(json::value());
            })
            .then([](pplx::task<json::value> previousTask) {
                try {
                    json::value jvalue = previousTask.get();
                    if (!jvalue.is_null() && jvalue.is_object()) {
                        string status = jvalue.at(U("status")).as_string();
                        if (strcmp(status.data(), "ok") == 0) {
                            cout << "nonce accepted by pool !!!!!" << endl;
                        } else {
                            cout << "nonce refused by pool :(:(:(" << endl;
                            cout << jvalue.to_string() << endl;
                            minerData.rejected += 1;
                        }
                    }
                }
                catch (http_exception const &e) {
                    cout << e.what() << endl;
                }
            });

}

static auto AS_SHA = std::unique_ptr<char[]>(new char[128]);

static void toHex(unsigned char *sha) {
    stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int j = 0; j < 64; ++j) {
        ss << std::setw(2) << (unsigned int) sha[j];
    }
    memcpy(AS_SHA.get(), ss.str().c_str(), strlen(ss.str().c_str()));
}

static void checkArgon(string *base, string *argon, string *nonce) {
    std::stringstream oss;
    oss << *base << *argon;
    auto sha = SHA512((const unsigned char *) oss.str().c_str(), strlen(oss.str().c_str()), NULL);
    sha = SHA512(sha, 64, NULL);
    sha = SHA512(sha, 64, NULL);
    sha = SHA512(sha, 64, NULL);
    sha = SHA512(sha, 64, NULL);
    sha = SHA512(sha, 64, NULL);
    toHex(sha);
    int a, b, c, d, e, f, g, h;
    std::stringstream aa;
    aa << std::hex << AS_SHA.get()[20] << AS_SHA.get()[21];
    aa >> a;
    aa.clear();
    aa << std::hex << AS_SHA.get()[30] << AS_SHA.get()[31];
    aa >> b;
    aa.clear();
    aa << std::hex << AS_SHA.get()[40] << AS_SHA.get()[41];
    aa >> c;
    aa.clear();
    aa << std::hex << AS_SHA.get()[46] << AS_SHA.get()[47];
    aa >> d;
    aa.clear();
    aa << std::hex << AS_SHA.get()[62] << AS_SHA.get()[63];
    aa >> e;
    aa.clear();
    aa << std::hex << AS_SHA.get()[80] << AS_SHA.get()[81];
    aa >> f;
    aa.clear();
    aa << std::hex << AS_SHA.get()[90] << AS_SHA.get()[91];
    aa >> g;
    aa.clear();
    aa << std::hex << AS_SHA.get()[110] << AS_SHA.get()[111];
    aa >> h;
    aa.clear();
    std::stringstream durationss;
    durationss << a << b << c << d << e << f << g << h;
    string duration = durationss.str();
    duration.erase(0, min(duration.find_first_not_of('0'), duration.size() - 1));
    mpz_t r;
    mpz_init(r);
    mpz_class result(duration);
    mpz_class diff(updateData.difficulty);
    mpz_tdiv_q(r, result.get_mpz_t(), diff.get_mpz_t());
    mpz_class l(updateData.limit);
    if (mpz_cmp(r, ZERO) > 0 && mpz_cmp(r, l.get_mpz_t()) <= 0) {
        mpz_cmp(r, BLOCK_LIMIT) < 0 ? minerData.blocks++ : minerData.shares++;
        gmp_printf("Submitting - %Zd - %s - %s\n", r, nonce->data(), argon->data());
        string argonTail = argon->substr(29);
        submit(&argonTail, nonce);
    }
    mpz_clear(r);
    aa.clear();
}

static int displayHeader = 0;

static void printStats() {
    if (displayHeader % 12 == 0) {
        cout << setw(20) << left << "ID"
             << setw(20) << left << "Batch avg duration"
             << setw(16) << left << "Hash rate /s"
             << setw(20) << left << "Total hashes"
             << setw(8) << left << "Shares"
             << setw(8) << left << "Blocks"
             << setw(8) << left << "Rejected"
             << endl;
    }
    cout << setw(20) << left << uniqid
         << setw(20) << left << ((1.0 * minerData.batchDuration) / minerData.batchCount)
         << setw(16) << left << ((minerData.hashes * 1000.0) / minerData.duration)
         << setw(20) << left << minerData.totalHashes
         << setw(8) << left << minerData.shares
         << setw(8) << left << minerData.blocks
         << setw(8) << left << minerData.rejected
         << endl;
    displayHeader++;
}


int main(int, const char *const *argv) {
    CommandLineParser<Arguments> parser = buildCmdLineParser();

    Arguments args;
    int ret = parser.parseArguments(args, argv);
    if (ret != 0) {
        return ret;
    }
    if (args.showHelp) {
        parser.printHelp(argv);
        return 0;
    }
    if (args.listDevices) {
        argon2::opencl::GlobalContext global;
        auto &devices = global.getAllDevices();
        for (std::size_t i = 0; i < devices.size(); i++) {
            auto &device = devices[i];
            std::cout << "Device #" << i << ": " << device.getName()
                      << std::endl;
        }
        return 0;
    }

    bool first = true;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int sec = (int) tv.tv_sec;
    int usec = (int) (tv.tv_usec % 0x100000);
    stringstream uniqids;
    uniqids << std::setfill('0') << std::setw(8) << std::hex << sec << std::setfill('0') << std::setw(5) << std::hex
            << usec;
    uniqid = uniqids.str();
    mpz_init(ZERO);
    mpz_set_str(BLOCK_LIMIT, "240", 10);

    http_client client(U("http://aropool.com"));
    updateInfoRequest(client);

    int batchSize = args.batchSize;
    auto saltBytes = new char[16];
    generateBytes(saltBytes, 16);

    argon2::opencl::GlobalContext global;
    auto &devices = global.getAllDevices();
    auto &device = devices[args.deviceIndex];
    cout << "using device " << device.getName() << endl;
    argon2::Type type = argon2::ARGON2_I;
    argon2::Version version = argon2::ARGON2_VERSION_13;
    argon2::opencl::ProgramContext progCtx(&global, {device}, type, version);

    argon2::Argon2Params params(32, saltBytes, 16, nullptr, 0, nullptr, 0, 4, 16384, 4);
    argon2::opencl::ProcessingUnit unit(&progCtx, &params, &device, batchSize, false, false);
    long hashes = 0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::string> nonces;
    std::vector<std::string> bases;
    std::vector<std::string> argons;
    while (true) {
        auto current = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
        if (time.count() > 5000) {
            minerData.duration += time.count();
            printStats();
            updateInfoRequest(client);
            minerData.batchDuration = 0;
            minerData.batchCount = 0;
            minerData.hashes = 0;
            minerData.duration = 0;
            start = std::chrono::high_resolution_clock::now();
        }
        nonces.clear();
        argons.clear();
        bases.clear();

        auto startBatch = std::chrono::high_resolution_clock::now();
        buildBatch(&nonces, &bases, batchSize);

        computeHash(&unit, &progCtx, &params, &bases, &argons, batchSize);

        for (int j = 0; j < nonces.size(); ++j) {
            checkArgon(&bases[j], &argons[j], &nonces[j]);
        }
        auto endBatch = std::chrono::high_resolution_clock::now();

        auto batchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endBatch - startBatch);
        minerData.batchDuration += batchDuration.count();
        minerData.batchCount += 1;
        minerData.hashes += batchSize;
        minerData.totalHashes += batchSize;
    }


}
