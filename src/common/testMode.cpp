#include <iostream>
#include <atomic>
#include <chrono>
#include <mutex>
#include <climits>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "../../include/testMode.h"
#include "../../include/stats.h"

using std::chrono::high_resolution_clock;

const int TEST_MODE_STATS_INTERVAL = 5;
const int TEST_MODE_BLOCK_CHANGE_RATE = 6;

struct TestModeInfo {
    bool enabled = false;
    argon2::time_point lastT = {};
    int nRounds = 0;
    BLOCK_TYPE blockType = BLOCK_GPU;
    bool testCPU = true;
    bool testGPU = true;
};

static TestModeInfo s_testMode;

void enableTestMode(bool testCPUBlocks, bool testGPUBlocks) {
    if (!testCPUBlocks && !testGPUBlocks)
        return;
    auto& tm = s_testMode;
    tm.enabled = true;
    tm.testCPU = testCPUBlocks;
    tm.testGPU = testGPUBlocks;
    if (!tm.testGPU) {
        tm.blockType = BLOCK_CPU;
    }
}

int testModeRoundLengthInSeconds() {
    return TEST_MODE_STATS_INTERVAL;
}

bool testMode() {
    return s_testMode.enabled;
}

BLOCK_TYPE testModeBlockType() {
    return s_testMode.blockType;
}

void updateTestMode(Stats &stats) {
    auto& tm = s_testMode;
    if (!tm.enabled)
        return;
    if (tm.lastT == argon2::time_point()) {
        tm.lastT = high_resolution_clock::now();
        stats.onBlockChange(tm.blockType);
        stats.printHeaderTestMode();
    }
    std::chrono::duration<float> duration =
        high_resolution_clock::now() - tm.lastT;
    if (duration.count() >= TEST_MODE_STATS_INTERVAL) {
        tm.nRounds++;

        stats.printStatsTestMode();

        bool blockChange = 
            tm.nRounds >= TEST_MODE_BLOCK_CHANGE_RATE &&
            tm.testCPU && tm.testGPU;
        if (blockChange) {
            tm.nRounds = 0;
            
            auto &bt = tm.blockType;
            bt = (bt == BLOCK_GPU) ? BLOCK_CPU : BLOCK_GPU;
            stats.onBlockChange(bt);
            
            std::cout << std::endl;
            stats.printHeaderTestMode();
        }

        tm.lastT = high_resolution_clock::now();
    }
}
