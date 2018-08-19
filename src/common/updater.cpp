//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include <iostream>
#include <thread>
#include "../../include/updater.h"

using namespace std;

#include <locale>
#include <codecvt>
#include <string>

#pragma warning(disable:4715)

void Updater::update() {
    stringstream paths;
    paths << "/mine.php?q=info&worker=" << *settings->getUniqid()
          << "&address=" << *settings->getPrivateKey()
          << "&hashrate=" << std::round(stats->getAvgHashRate());

    http_request req(methods::GET);
    req.headers().set_content_type(L"application/json");

#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring _paths = converter.from_bytes(paths.str());
#else
    std::string _paths = paths.str();
#endif

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
        wstring status = value->at(L"status").as_string();
        if (status == L"ok") {
            json::value jsonData = value->at(L"data");
            if (jsonData.is_object()) {
                wstring difficulty = jsonData.at(L"difficulty").as_string();
                wstring block = jsonData.at(L"block").as_string();
                wstring public_key = jsonData.at(L"public_key").as_string();
                int limit = jsonData.at(L"limit").as_integer();
                string limitAsString = std::to_string(limit);

                std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
                std::string narrow_status = converter.to_bytes(status);
                std::string narrow_block = converter.to_bytes(block);
                std::string narrow_diff = converter.to_bytes(difficulty);
                std::string narrow_public_key = converter.to_bytes(public_key);

                uint32_t argon_memory = (uint32_t)jsonData.at(L"argon_mem").as_integer();
                uint32_t argon_threads = (uint32_t)jsonData.at(L"argon_threads").as_integer();
                uint32_t argon_time = (uint32_t)jsonData.at(L"argon_time").as_integer();
                uint32_t height = jsonData.at(L"height").as_integer();

                wstring recommendation = jsonData.at(L"recommendation").as_string();
                std::string narrow_recommendation = converter.to_bytes(recommendation);
                BLOCK_TYPE blockType;
                if (narrow_recommendation != "mine") {
                    blockType = BLOCK_MASTERNODE;
                }
                else {
                    blockType = (argon_threads != 1) ? BLOCK_GPU : BLOCK_CPU;
                }

                if (data == NULL || data->isNewBlock(&narrow_block)) {
                    data = new MinerData(
                        narrow_status, narrow_diff, limitAsString, narrow_block, narrow_public_key,
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

#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring _poolAddress = converter.from_bytes(*(ms->getPoolAddress()));
#else
    std::string _poolAddress = _poolAddress;
#endif

    client = new http_client(_poolAddress, config);
}
