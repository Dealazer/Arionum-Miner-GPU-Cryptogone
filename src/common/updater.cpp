//
// Created by guli on 31/01/18.
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

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring pathsw = converter.from_bytes(paths.str());

    req.set_request_uri(pathsw.data());
    
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
                if (data == NULL || data->isNewBlock(&narrow_block)) {
                    data = new MinerData(narrow_status, narrow_diff, limitAsString, narrow_block, narrow_public_key);
                    if (gMiningStarted) {
                        cout << endl << "-- NEW BLOCK FOUND --" << endl << *data << endl;
                    }
                    stats->blockChange();
                }
            }
        } else {
            cerr << "Unable to get result" << endl;
        }
    }
}

MinerData *Updater::getData() {
    std::lock_guard<std::mutex> lg(mutex);
    return data;
}

Updater::Updater(Stats *s, MinerSettings *ms) : stats(s), settings(ms) {
    http_client_config config;
    utility::seconds timeout(2);
    config.set_timeout(timeout);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring poolAddressW = converter.from_bytes(*(ms->getPoolAddress()));

    client = new http_client(poolAddressW, config);
}
