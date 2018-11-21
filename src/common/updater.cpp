//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//
#include "../../include/updater.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <thread>
#include <boost/algorithm/string.hpp>

const int UPDATER_REQ_TIMEOUT_SECONDS = 2;

const double FIRST_HASHRATE_SEND_N_MINUTES = 0.5;
const double NEXT_HASHRATE_SENDS_N_MINUTES = 5.0;

// some pools ignore hashrates reporting when the cpu block hashrate is zero
// ex: if miner starts on a GPU block, gpu hashrate will be ignored until switching to a CPU block
// so in this case, we send a dummy CPU hashrate (1.0) to make sure miner is visible on pool stats
const double DEFAULT_CPU_HASHRATE = 1.0;

bool s_miningReady = false;

void Updater::update() {
    static bool hrCPUSentAtLeastOneTime = false;
    static long long nHashRateSends = 0;
    static auto start = std::chrono::system_clock::now();

    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    auto timeSinceLastHrUpdateMs = duration.count();
    double hrSendRate_mins = (nHashRateSends == 0) ? 
        FIRST_HASHRATE_SEND_N_MINUTES : NEXT_HASHRATE_SENDS_N_MINUTES;
    auto hrCPU = std::round(stats.getAvgHashrate(BLOCK_CPU));
    bool sendHashrate =
        ((double)timeSinceLastHrUpdateMs >= hrSendRate_mins * 60.0 * 1000.0) ||
        (hrCPU > 0 && !hrCPUSentAtLeastOneTime);

    std::stringstream paths;
    paths << "/mine.php?q=info&worker=" << settings.uniqueID()
        << "&address=" << settings.privateKey();

    if (sendHashrate) {
        if (hrCPU > 0)
            hrCPUSentAtLeastOneTime = true;
        else
            hrCPU = DEFAULT_CPU_HASHRATE;
        paths << "&hashrate=" << hrCPU
              << "&hrgpu=" << std::round(stats.getAvgHashrate(BLOCK_GPU));
        if (nHashRateSends != 0)
            start = now;
        nHashRateSends++;
    }

    auto _paths = toUtilityString(paths.str());
    http_request req(methods::GET);
    req.headers().set_content_type(U("application/json"));
    req.set_request_uri(_paths.data());

    client->request(req)
    .then([](web::http::http_response response)
    {
        if (response.status_code() == status_codes::OK) {
            response.headers().set_content_type(U("application/json"));
            return response.extract_json();
        }
        return pplx::task_from_result(json::value());
    })
    .then([this](pplx::task<json::value> previousTask) -> void 
    {
        try {
            const json::value &value = previousTask.get();
            processResponse(&value);
        }
        catch (const web::http::http_exception & e) {
            std::cout << "-- Updater http_exception => " << e.what() << std::endl;
            newClient();
        }
        catch (const web::json::json_exception & e) {
            std::cout << "-- Updater json_exception => " << e.what() << std::endl;
        }
        catch (const std::exception & e) {
            std::cout << "-- Updater exception => " << e.what() << std::endl;
        }
    }).wait();
}

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
        MinerData roundData = getData();
        stats.beginRound(roundData.getBlockType());
        while (true) {
            auto start = std::chrono::system_clock::now();
            update();
            while (1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                std::chrono::duration<float> duration = std::chrono::system_clock::now() - start;
                MinerData curData = getData();
                bool blockTypeChanged = curData.getBlockType() != roundData.getBlockType();
                bool roundFinished = (duration.count() >= POOL_UPDATE_RATE_SECONDS || blockTypeChanged);
                if (roundFinished) {
                    stats.endRound();
                    if (!blockTypeChanged) {
                        stats.printMiningStats(
                            roundData,
                            settings.useLastHashrateInsteadOfRoundAvg(),
                            settings.canMineBlock(roundData.getBlockType()));
                    }
                    roundData = getData();
                    stats.beginRound(roundData.getBlockType());
                    break;
                }
                if (refreshCount > 0) {
                    std::cout << "-- refreshing pool info" << std::endl;
                    update();
                    refreshCount = 0;
                }
            }
        }
    }
    catch (const std::exception & e) {
        std::cout << "Exception in update thread: " << e.what() << std::endl;
        exit(1);
    }
}

void Updater::requestRefresh() {
    refreshCount++;
}

void Updater::processResponse(const json::value *value) {
    if (value->is_null() || !value->is_object()) {
        throw std::logic_error("Failed to get work from pool (empty json)");
    }

    std::string status = toString(value->at(U("status")).as_string());
    if (status == "ok") {
        json::value jsonData = value->at(U("data"));

        if (jsonData.is_object()) {
            std::string difficulty = toString(jsonData.at(U("difficulty")).as_string());
            std::string block = toString(jsonData.at(U("block")).as_string());
            std::string public_key = toString(jsonData.at(U("public_key")).as_string());

            int limit = jsonData.at(U("limit")).as_integer();
            std::string limitAsString = std::to_string(limit);

            uint32_t argon_memory = (uint32_t)jsonData.at(U("argon_mem")).as_integer();
            uint32_t argon_threads = (uint32_t)jsonData.at(U("argon_threads")).as_integer();
            uint32_t argon_time = (uint32_t)jsonData.at(U("argon_time")).as_integer();
            uint32_t height = jsonData.at(U("height")).as_integer();

            std::string recommendation = toString(jsonData.at(U("recommendation")).as_string());
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
                        std::cout << std::endl << "-- NEW BLOCK --" << std::endl << *data;
                    }
                    stats.blockChange(data->getBlockType());
                }
            }
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

Updater::Updater(Stats &s, MinerSettings ms) : 
    stats(s), data{}, settings(ms), client{}, refreshCount{} {
    newClient();
}

void Updater::newClient() {
    http_client_config config;
    config.set_timeout(utility::seconds(UPDATER_REQ_TIMEOUT_SECONDS));
    auto poolAddress = toUtilityString(settings.poolAddress());
    client = new http_client(poolAddress, config);
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
