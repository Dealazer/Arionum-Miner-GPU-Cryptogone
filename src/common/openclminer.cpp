//
// Created by guli on 31/01/18.
//

#include <iostream>
#include "../../include/openclminer.h"

using namespace std;

OpenClMiner::OpenClMiner(Stats *s, MinerSettings *ms, MinerData *d, Updater *u, size_t *deviceIndex)
        : Miner(s, ms, d, u) {
    generateBytes(saltBase64, 32, saltBuffer, 16);
    string salt(saltBase64);
    global = new argon2::opencl::GlobalContext();
    cout << "MINER AAAAAAAAAAAAAAA" << endl;
/*
    auto &devices = global->getAllDevices();
    device = &devices[*deviceIndex];
    cout << "using device " << device->getName() << endl;
    cout << "using salt " << salt << endl;
    progCtx = new argon2::opencl::ProgramContext(global, {*device}, type, version);
    params = new argon2::Argon2Params(32, saltBase64, 16, nullptr, 0, nullptr, 0, 4, 16384, 4);

    cout << "MINER BBBBBBBBBBBBBBBBBBBBBBBBB" << params->getTimeCost() << endl;

    unit = new argon2::opencl::ProcessingUnit(progCtx, params, device, 1, false, false);
    */
    argon2::opencl::GlobalContext global;
    auto &devices = global.getAllDevices();
    auto &device = devices[*deviceIndex];
    cout << "using device " << device.getName() << endl;
    argon2::Type type = argon2::ARGON2_I;
    argon2::Version version = argon2::ARGON2_VERSION_13;
    argon2::opencl::ProgramContext progCtx(&global, {device}, type, version);

    argon2::Argon2Params params(32, saltBase64, 16, nullptr, 0, nullptr, 0, 4, 16384, 4);
    argon2::opencl::ProcessingUnit unit(&progCtx, &params, &device, *settings->getBatchSize(), false, false);

    cout << "MINER READYYYYYYYYY" << endl;
}

void OpenClMiner::computeHash() {
    size_t size = *settings->getBatchSize();
    for (size_t j = 0; j < size; ++j) {
        std::string data = bases.at(j);
        unit->setPassword(j, data.data(), data.length());
    }
    unit->beginProcessing();
    unit->endProcessing();
    auto buffer = std::unique_ptr<std::uint8_t[]>(new std::uint8_t[32]);
    for (size_t j = 0; j < size; ++j) {
        unit->getHash(j, buffer.get());
        char *openClEnc = encode(buffer.get(), 32);
        argons.push_back(std::string(openClEnc));
    }
}