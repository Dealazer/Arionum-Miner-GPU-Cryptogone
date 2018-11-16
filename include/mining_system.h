#pragma once

#include "miner.h"
#include "stats.h"
#include "updater.h"
#include "testMode.h"
#include "miners_stats.h"
#include "perfscope.h"

#include <chrono>
#include <vector>
#include <thread>

const size_t MAX_MINERS_PER_DEVICE = 256;

struct DeviceConfig {
    uint32_t deviceIndex;
    uint32_t nMiners;
    uint32_t gpuBatchSize;
    argon2::OPT_MODE cpuBlocksOptimizationMode;
};

template <class CONTEXT, class MINING_DEVICE, class MINER>
class MiningSystem {
public:
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

            cout << endl;
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

            if (!miner->generateNonces())
                continue;

            miner->launchGPUTask();

            minerIdle[i] = false;
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
                PerfScope p("processResults()");

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
