//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#ifndef ARIONUM_GPU_MINER_UPDATER_H
#define ARIONUM_GPU_MINER_UPDATER_H

#include "stats.h"
#include "minerdata.h"
#include "minersettings.h"
#include <cpprest/http_client.h>
#include <mutex>
#include <locale>
#include <codecvt>

using namespace web;
using namespace web::http;
using namespace web::http::client;

std::string toString(const utility::string_t &s);
utility::string_t toUtilityString(const std::string &s);

const int POOL_UPDATE_RATE_SECONDS = 3;

class Updater {

protected:
    Stats *stats{};
    MinerData *data{};
    MinerSettings *settings{};
    http_client *client{};

    std::mutex mutex;

public:
    void update();

    void start();

    explicit Updater(Stats *s, MinerSettings *ms);

    void processResponse(const json::value *value);

    MinerData getData();
};

#endif //ARIONUM_GPU_MINER_UPDATER_H
