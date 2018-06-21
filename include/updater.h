//
// Created by guli on 31/01/18.
//

#ifndef ARIONUM_GPU_MINER_UPDATER_H
#define ARIONUM_GPU_MINER_UPDATER_H

#include "stats.h"
#include "minerdata.h"
#include "minersettings.h"
#include <cpprest/http_client.h>
#include <mutex>

using namespace web;
using namespace web::http;
using namespace web::http::client;

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

    MinerData *getData();
};

#endif //ARIONUM_GPU_MINER_UPDATER_H
