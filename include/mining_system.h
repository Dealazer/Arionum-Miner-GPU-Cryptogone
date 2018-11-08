#pragma once

#include <chrono>
#include <vector>
#include "miner.h"
#include "stats.h"
#include "updater.h"
#include "testMode.h"
#include "miners_stats.h"
#include "perfscope.h"

const size_t MAX_MINERS_PER_DEVICE = 256;

template <class CONTEXT, class MINING_DEVICE, class MINER>
class MiningSystem {
public:
    struct DeviceConfig {
        uint32_t deviceIndex;
        uint32_t nMiners;
        uint32_t gpuBatchSize;
        argon2::OPT_MODE cpuBlocksOptimizationMode;
    };

    MiningSystem(
        const std::vector<DeviceConfig> & deviceConfigs,
        std::unique_ptr<IAroNonceProvider> nonceProvider,
        std::unique_ptr<IAroResultsProcessor> resultsProcessor,
        Stats &stats) :
        startTime(std::chrono::high_resolution_clock::now()),
        globalContext(new CONTEXT()),
        nonceProvider(std::move(nonceProvider)),
        resultsProcessor(std::move(resultsProcessor)),
        miningDevices{},
        miners{},
        devicesMiners{},
        minerIdle{},
        minerStartT(MAX_MINERS_PER_DEVICE),
        stats(stats) {
    }

    void createMiners(const std::vector<DeviceConfig> & devicesConfigs) {
        miners.clear();
        miningDevices.clear();

        int nDevices = 0;
        auto &devices = globalContext->getAllDevices();
        for (int i = 0; i < devicesConfigs.size(); i++) {
            auto deviceIndex = devicesConfigs[i].deviceIndex;
            if (deviceIndex >= devices.size()) {
                cout << endl;
                cout << "--- Device " << deviceIndex << " does not exist, skipping it" << endl;
                cout << endl;
                continue;
            }

            // skip CPU devices
            if (!strcmp(API_NAME, "OPENCL")) {
                auto devInfo = devices[deviceIndex].getInfo();
                if (devInfo.find("Type: GPU") == std::string::npos) {
                    continue;
                }
            }

            cout << "--- Device " << deviceIndex
                << ", " << devices[deviceIndex].getName() << " ---" << endl;
            uint32_t nMiners = devicesConfigs[i].nMiners;
            uint32_t gpuBatchSize = devicesConfigs[i].gpuBatchSize;
            miningDevices.emplace_back(
                AroMiningDeviceFactory::create<MINING_DEVICE>(
                    deviceIndex, nMiners, gpuBatchSize));
            devicesMiners.push_back({});
            auto & miningDevice = *miningDevices.back();
            for (uint32_t j = 0; j < nMiners; ++j) {
                auto pMiner = new MINER(
                    miningDevice.programContext(),
                    miningDevice.queue(j),
                    miningDevice.memConfig(j),
                    { *nonceProvider, *resultsProcessor },
                    devicesConfigs[i].cpuBlocksOptimizationMode);
                miners.emplace_back((AroMiner*)pMiner);
                devicesMiners.back().push_back(miners.size() - 1);
                cout << "miner " << j << " : " << miners.back()->describe() << endl;
            }
            cout << endl;
        }

        minerIdle = std::vector<bool>(miners.size(), true);
    }

    void miningLoop() {
        while (true) {
            bool ok = feedMiners();
            if (!ok)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

            int nIdle = processMinersResults();
            if (nIdle == 0) {
                //std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    const AroMiner & getMiner(int index) {
        return *miners[index];
    }

protected:
    t_time_point startTime;
    std::unique_ptr<CONTEXT> globalContext;
    std::unique_ptr<IAroNonceProvider> nonceProvider;
    std::unique_ptr<IAroResultsProcessor> resultsProcessor;
    std::vector<std::unique_ptr<MINING_DEVICE>> miningDevices;
    std::vector<std::unique_ptr<AroMiner>> miners;
    std::vector<vector<size_t>> devicesMiners;
    std::vector<bool> minerIdle;
    std::vector<t_time_point> minerStartT;
    Stats &stats;

    size_t minerDeviceIndex(size_t minerIndex) {
        for (size_t i = 0; i < devicesMiners.size(); i++)
            for (auto &it : devicesMiners[i])
                if (it == minerIndex)
                    return i;
        throw std::logic_error("cannot find miner device !");
        return 0;
    }

    bool allDeviceMinersIdle(size_t deviceIndex) {
        for (auto &minerIndex : devicesMiners[deviceIndex]) {
            if (!minerIdle[minerIndex])
                return false;
        }
        return true;
    }

    bool canStartMiner(size_t deviceIndex, BLOCK_TYPE blockType) {
        for (auto minerID : devicesMiners[deviceIndex]) {
            if (!minerIdle[minerID] && miners[minerID]->taskBlockType() != blockType)
                return false;
        }
        return true;
    }

    void checkShowStartMessage(Stats &stats) {
        static bool s_miningStartedMsgShown = false;
        if (!s_miningStartedMsgShown) {
            stats.printTimePrefix();
            cout <<
                (testMode() ?
                    "--- Start Testing ---" : "--- Start Mining ---")
                << endl;
            s_miningStartedMsgShown = true;
        }
    }

    bool feedMiners() {
        for (int i = 0; i < miners.size(); i++) {
            t_time_point now = std::chrono::high_resolution_clock::now();
            if (minerIdle[i] == false)
                continue;

            checkShowStartMessage(stats);

            auto &miner = miners[i];
            if (!miner->updateNonceProvider())
                continue;

            minerStatsOnNewTask(*miner, i, (int)miners.size(), now);

            if (!canStartMiner(minerDeviceIndex(i), miner->providerBlockType()))
                continue;

#if DEBUG_DURATIONS
            std::chrono::duration<float> T = high_resolution_clock::now() - s_start;
            printf("T=%4.3f miner %d, start %s task\n", T.count(), i, blockTypeName(s_miners[i]->getCurrentBlockType()).c_str());
            if (s_minerStartT[i] == t_time_point()) {
                s_minerStartT[i] = high_resolution_clock::now();
            }
#endif

            if (!miner->generateNonces())
                continue;

            miner->launchGPUTask();

            minerIdle[i] = false;

#if 0
            std::chrono::duration<float> duration = high_resolution_clock::now() - s_minerStartT[i];
            printf("miner %d, new task set in %.2fms\n",
                i, duration.count() * 1000.f);
#endif
        }
        return true;
    }

    int processMinersResults() {
        int nIdle = 0;
        for (int i = 0; i < miners.size(); i++) {
            if (minerIdle[i]) {
                nIdle++;
                continue;
            }
            if (miners[i]->resultsReady()) {
#if DEBUG_DURATIONS
                if (s_minerStartT[i] != t_time_point()) {
                    std::chrono::duration<float> T = high_resolution_clock::now() - start;
                    std::chrono::duration<float> duration = high_resolution_clock::now() - minerStartT[i];
                    printf("T=%4.3f miner %d, %d %s hashes in %.2fms => %.1f Hs\n",
                        T.count(),
                        i,
                        miners[i]->getNbHashesPerIteration(),
                        blockTypeName(miners[i]->getCurrentBlockType()).c_str(),
                        duration.count() * 1000.f,
                        (float)miners[i]->getNbHashesPerIteration() / duration.count());
                }
                s_minerStartT[i] = high_resolution_clock::now();
#endif
                auto result = miners[i]->processResults();
                stats.addHashes(result.nHashes);
                minerStatsOnTaskEnd(i, result.valid);
                minerIdle[i] = true;

                if (testMode() && (result.nGood != result.nHashes)) {
                    std::cout
                        << "Warning: found invalid argon results in batch !"
                        << " (" << (result.nHashes - result.nGood) << " / " << result.nHashes << ")"
                        << std::endl;
                }
            }
        }
        return nIdle;
    }
};




//int deviceListItem = -1;
//for (const auto &it : deviceIndexList) {
//    deviceListItem++;
//    if (it >= devices.size()) {
//        cout << endl;
//        cout << "--- Device " << it << " does not exist, skipping it" << endl;
//        cout << endl;
//        continue;
//    }

//    // skip CPU devices (only needed for OpenCL)
//    if (!strcmp(API_NAME, "OPENCL")) {
//        auto devInfo = devices[it].getInfo();
//        if (devInfo.find("Type: GPU") == std::string::npos) {
//            //std::cout << std::endl;
//            //std::cout 
//            //  << "--- Device " << it << " is not a GPU, skipping it" 
//            //  << std::endl << devices[it].getName() << std::endl;
//            continue;
//        }
//    }

//    uint32_t deviceIndex = it;
//    uint32_t nThreads = nTasksPerDeviceList.back();
//    if (deviceListItem < nTasksPerDeviceList.size()) {
//        nThreads = nTasksPerDeviceList[deviceListItem];
//    }

//    uint32_t nGPUBatches = gpuBatchSizePerDeviceList.back();
//    if (deviceListItem < gpuBatchSizePerDeviceList.size()) {
//        nGPUBatches = gpuBatchSizePerDeviceList[deviceListItem];
//    }

//    cout << "--- Device " << deviceIndex << ", " << devices[it].getName() << " ---" << endl;
//    miningDevices.emplace_back(
//        AroMiningDeviceFactory::create<MINING_DEVICE>(
//            deviceIndex, nThreads, nGPUBatches));

//    auto & miningDevice = *miningDevices.back();
//    devicesMiners.push_back({});
//    for (uint32_t j = 0; j < nThreads; ++j) {
//        devicesMiners.back().push_back(miners.size());
//        miners.push_back(new MINER(
//            miningDevice.programContext(),
//            miningDevice.queue(j),
//            miningDevice.memConfig(j),
//            { *nonceProvider, *resultsProcessor },
//            args.cpuBlocksOptimizationMode));
//        cout << "miner " << j << " : " << miners.back()->describe() << endl;
//    }
//    cout << endl;
//}

//minerIdle = std::vector<bool>(s_miners.size(), true);

// -u option
//if (args.allDevices) {
//    args.deviceIndexList.clear();
//    for (int i = 0; i < devices.size(); ++i) {
//        args.deviceIndexList.push_back(i);
//    }
//}

//
//if (args.deviceIndexList.size() == 0) {
//    std::cout << "Error: no device found, aborting" << std::endl;
//    exit(1);
//}

// create miners
// cout << endl;