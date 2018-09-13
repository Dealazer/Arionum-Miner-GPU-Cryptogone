//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include <iostream>
#include <iomanip>
#include <thread>
#include "../../include/updater.h"

using namespace std;

#include <string>

#pragma warning(disable:4715)

//#define DEBUG_HASHRATE_SENDS

void Updater::update() {
    // start building path
    stringstream paths;
    paths << "/mine.php?q=info&worker=" << *settings->getUniqid()
          << "&address=" << *settings->getPrivateKey();

    // see if we need to send hashrates (pools recommend every 10 minutes)
    static auto start = std::chrono::system_clock::now();    
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    auto timeSinceLastHrUpdateMs = duration.count();

    // send first hashrates after 30s
    static long long nHashRateSends = 0;
    double hrSendRate_mins = (nHashRateSends == 0) ? 0.5 : 5.0;

    // as soon as we get the first cpu hashrate value, set rate to zero to force send it
    auto hrCPU = std::round(stats->getAvgHashrate(BLOCK_CPU));
    static bool s_hrCpuOkOneTime = false;
    if (hrCPU > 0 &&
        !s_hrCpuOkOneTime) {
        hrSendRate_mins = 0;
        s_hrCpuOkOneTime = true;
#ifdef DEBUG_HASHRATE_SENDS
        cout << "##### FIRST CPU HASHRATE" << endl;
#endif
    }
    else {
        if (hrCPU <= 0)
            hrCPU = 1;
    }

    // send hashrates when update period reached
    if ((double)timeSinceLastHrUpdateMs >= hrSendRate_mins * 60.0 * 1000.0) {
        paths << "&hashrate=" << hrCPU
              << "&hrgpu=" << std::round(stats->getAvgHashrate(BLOCK_GPU));
        if (nHashRateSends != 0) {
            start = now;
        }
        nHashRateSends++;
#ifdef DEBUG_HASHRATE_SENDS
        cout << "##### SENDING HASHRATES nHashRateSends=" << nHashRateSends << endl;
        cout << paths.str() << endl;
#endif
    }

    // perform request with cpp rest
    http_request req(methods::GET);
    req.headers().set_content_type(U("application/json"));
    auto _paths = toUtilityString(paths.str());
    req.set_request_uri(_paths.data());
    client->request(req)
            .then([](http_response response) {
                try {
                    if (response.status_code() == status_codes::OK) {
                        response.headers().set_content_type(U("application/json"));
                        return response.extract_json();
                    }
                    return pplx::task_from_result(json::value());
                } catch (http_exception const &e) {
                    cerr << e.what() << endl;
                } catch (web::json::json_exception const &e) {
                    cerr << e.what() << endl;
                }
            })
            .then([this](pplx::task<json::value> previousTask) {
                try {
                    const json::value &value = previousTask.get();
                    processResponse(&value);
                } catch (http_exception const &e) {
                    cerr << e.what() << endl;
                } catch (web::json::json_exception const &e) {
                    cerr << e.what() << endl;
                }
            })
            .wait();
}

extern bool s_miningReady;

void Updater::start() {
    try {
        // wait for first  pool response
        while (true) {
            update();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            bool dataReady = false;
            {
                std::lock_guard<std::mutex> lg(mutex);
                dataReady = data && data->isValid();
            }
            if (dataReady) {
                while (!s_miningReady) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                break;
            }
            else {
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }

        // loop
        while (true) {
            MinerData newData = getData();
            stats->beginRound(newData.getBlockType());
            {
                update();
                std::this_thread::sleep_for(std::chrono::seconds(POOL_UPDATE_RATE_SECONDS));
            }
            stats->endRound();
            cout << *stats;
        }
    }
    catch (exception e) {
        std::cout << "Exception in update thread: " << e.what() << std::endl;
        exit(1);
    }
}

void Updater::processResponse(const json::value *value) {
    if (!value->is_null() && value->is_object()) {
        string status = toString(value->at(U("status")).as_string());
        if (status == "ok") {
            json::value jsonData = value->at(U("data"));

            if (jsonData.is_object()) {
                string difficulty = toString(jsonData.at(U("difficulty")).as_string());
                string block = toString(jsonData.at(U("block")).as_string());
                string public_key = toString(jsonData.at(U("public_key")).as_string());

                int limit = jsonData.at(U("limit")).as_integer();
                string limitAsString = std::to_string(limit);

                uint32_t argon_memory = (uint32_t)jsonData.at(U("argon_mem")).as_integer();
                uint32_t argon_threads = (uint32_t)jsonData.at(U("argon_threads")).as_integer();
                uint32_t argon_time = (uint32_t)jsonData.at(U("argon_time")).as_integer();
                uint32_t height = jsonData.at(U("height")).as_integer();

                string recommendation = toString(jsonData.at(U("recommendation")).as_string());
                BLOCK_TYPE blockType;
                if (recommendation != "mine") {
                    blockType = BLOCK_MASTERNODE;
                }
                else {
                    blockType = (argon_threads != 1) ? BLOCK_GPU : BLOCK_CPU;
                }

                {
                    std::lock_guard<std::mutex> lg(mutex);

                    if (data == NULL || data->isNewBlock(&block)) {
                        if (data)
                            delete data;
                        data = new MinerData(
                            status, difficulty, limitAsString, block, public_key,
                            height,
                            argon_memory,
                            argon_threads,
                            argon_time,
                            blockType
                        );

                        if (s_miningReady) {
                            cout << endl << "-- NEW BLOCK --" << endl << *data;
                        }
                        stats->blockChange(data->getBlockType());
                    }
                }
            }
        } else {
            cerr << "Unable to get result" << endl;
        }
    }
}

MinerData Updater::getData() {
    std::lock_guard<std::mutex> lg(mutex);
    if (!data) {
        return {};
    }
    return *data;
}

Updater::Updater(Stats *s, MinerSettings *ms) : stats(s), settings(ms) {
    http_client_config config;
    utility::seconds timeout(2);
    config.set_timeout(timeout);

    auto _poolAddress = toUtilityString(*ms->getPoolAddress());
    client = new http_client(_poolAddress, config);
}

#ifdef _WIN32
std::string toString(const utility::string_t &s) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.to_bytes(s);
}

utility::string_t toUtilityString(const std::string &s) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(s);
}
#else
std::string toString(const utility::string_t &s) {
    return s;
}

utility::string_t toUtilityString(const std::string &s) {
    return s;
}
#endif
