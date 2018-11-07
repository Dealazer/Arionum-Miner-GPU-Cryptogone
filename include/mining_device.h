#pragma once

#include <vector>
#include <memory>
#include "aro_tools.h"

class IMiningDevice {
public:
    virtual void initializeDevice(uint32_t deviceIndex) = 0;
    virtual void* newQueue() = 0;
    virtual void* newBuffer(size_t size) = 0;
    virtual void writeBuffer(void* buf, const void* src, size_t size) = 0;
    virtual ~IMiningDevice() {};
};

template<typename QUEUE, typename BUFFER>
class MiningDevice : public IMiningDevice {
public:
    QUEUE& queue(uint32_t i) {
        return *queues[i];
    }

    BUFFER& buffer(uint32_t i) {
        return *buffers[i];
    }
protected:
    std::vector<std::unique_ptr<BUFFER>> buffers;
    std::vector<std::unique_ptr<QUEUE>> queues;
};

template<typename QUEUE, typename BUFFER>
class AroMiningDevice : public MiningDevice<QUEUE, BUFFER> {
public:
    const argon2::MemConfig& memConfig(uint32_t threadIndex) {
        return minersConfigs[threadIndex];
    }
    friend class AroMiningDeviceFactory;

protected:
    std::vector<argon2::MemConfig> minersConfigs;
};

class AroMiningDeviceFactory {
public:
    template<typename T>
    static T* create(
        std::uint32_t deviceIndex,
        std::uint32_t nTasks,
        std::uint32_t batchSizeGPU) {
        auto pDevice = new T();
        pDevice->initializeDevice(deviceIndex);
        pDevice->minersConfigs = configureForAroMining(*pDevice, nTasks, batchSizeGPU);
        return pDevice;
    }

protected:
    static std::vector<argon2::MemConfig> configureForAroMining(
        IMiningDevice& device,
        uint32_t nTasks, uint32_t batchSizeGPU);
};
