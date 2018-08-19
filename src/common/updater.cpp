//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include <iostream>
#include <thread>
#include "../../include/updater.h"

using namespace std;

#include <string>

#pragma warning(disable:4715)

void Updater::update() {
    stringstream paths;
    paths << "/mine.php?q=info&worker=" << *settings->getUniqid()
          << "&address=" << *settings->getPrivateKey()
          << "&hashrate=" << std::round(stats->getAvgHashRate());

    http_request req(methods::GET);
    req.headers().set_content_type(L"application/json");

    auto _paths = toUtilityString(paths.str());
    req.set_request_uri(_paths.data());
    
    client->request(req)
            .then([](http_response response) {
                try {
                    if (response.status_code() == status_codes::OK) {
                        response.headers().set_content_type(L"application/json");
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

extern bool gMiningStarted;

void Updater::start() {
    while (true) {
        if (gMiningStarted) {
            stats->newRound();
            if (stats->getRounds() > 1) {
                cout << *stats << endl;
            }
        }
        update();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

void Updater::processResponse(const json::value *value) {
    std::lock_guard<std::mutex> lg(mutex);
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

                if (data == NULL || data->isNewBlock(&block)) {
                    data = new MinerData(
                        status, difficulty, limitAsString, block, public_key,
                        height,
                        argon_memory,
                        argon_threads,
                        argon_time,
                        blockType
                    );
                    if (gMiningStarted) {
                        cout << endl << "-- NEW BLOCK --" << endl << *data << endl;
                    }
                    stats->blockChange();
                }
            }
        } else {
            cerr << "Unable to get result" << endl;
        }
    }
}

MinerData Updater::getData() {
    std::lock_guard<std::mutex> lg(mutex);
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
