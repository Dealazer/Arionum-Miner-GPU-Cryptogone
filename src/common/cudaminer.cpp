//
// Created by guli on 31/01/18.
//

#include <iostream>
#include "../../include/cudaminer.h"

using namespace std;

CudaMiner::CudaMiner(Stats *s, MinerSettings *ms, Updater *u, size_t *deviceIndex) : Miner(s, ms, u) {
    global = new argon2::cuda::GlobalContext();
    auto &devices = global->getAllDevices();
    device = &devices[*deviceIndex];
    cout << "using device " << device->getName() << endl;
    cout << "using salt " << salt << endl;
    progCtx = new argon2::cuda::ProgramContext(global, {*device}, type, version);
    params = new argon2::Argon2Params(32, salt.data(), 16, nullptr, 0, nullptr, 0, 4, 16384, 4);
    unit = new argon2::cuda::ProcessingUnit(progCtx, params, device, *settings->getBatchSize(), false, false);
}

void CudaMiner::computeHash() {
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
        string encodedArgon(openClEnc);
        argons.push_back(encodedArgon);
    }
}