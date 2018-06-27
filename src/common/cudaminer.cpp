//
// Created by guli on 31/01/18.
//

#include <iostream>
#include "../../include/cudaminer.h"
#include "../../argon2-gpu/include/argon2-cuda/processingunit.h"
#include "../../argon2-gpu/include/argon2-cuda/programcontext.h"
#include "../../argon2-gpu/include/argon2-cuda/device.h"
#include "../../argon2-gpu/include/argon2-cuda/globalcontext.h"
#include "../../argon2-gpu/include/argon2-cuda/cudaexception.h"

using namespace std;

static void setCudaDevice(int deviceIndex)
{	
    int currentIndex = -1;
    argon2::cuda::CudaException::check(cudaGetDevice(&currentIndex));
    if (currentIndex != deviceIndex) {
        argon2::cuda::CudaException::check(cudaSetDevice(deviceIndex));
    }
}

CudaMiner::CudaMiner(Stats *s, MinerSettings *ms, Updater *u, size_t *deviceIndex) : Miner(s, ms, u) {
    global = new argon2::cuda::GlobalContext();
    auto &devices = global->getAllDevices();

    if (*deviceIndex >= devices.size()) {
        cout << endl << "!!! Warning !!! invalid device index: -d " << *deviceIndex <<", will use device 0 instead" << endl << endl;
        *deviceIndex = 0;
    }

    device = &devices[*deviceIndex];
    cout << "using device " << *deviceIndex << " - " << device->getName() << endl;
    cout << "using salt " << salt << endl;

    setCudaDevice(device->getDeviceIndex());

    progCtx = new argon2::cuda::ProgramContext(global, {*device}, type, version);
    params = new argon2::Argon2Params(32, salt.data(), 16, nullptr, 0, nullptr, 0, 1, 524288, 1);

    try {
        unit = new argon2::cuda::ProcessingUnit(progCtx, params, device, *settings->getBatchSize(), false, false);
    }
    catch (const std::exception& e) {
        cout << "Error: exception while creating cudaminer unit: " << e.what() << ", try to reduce batch size (-b parameter), exiting now :-(" << endl;
        exit(1);
    }
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