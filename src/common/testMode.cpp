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

using namespace std;
using std::chrono::high_resolution_clock;

typedef std::chrono::time_point<high_resolution_clock> t_time_point;

const int TEST_MODE_STATS_INTERVAL = 5;
const int TEST_MODE_BLOCK_CHANGE_RATE = 4;

struct TestModeInfo {
    bool enabled = false;
    t_time_point lastT = {};
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
    if (tm.lastT == t_time_point()) {
        tm.lastT = high_resolution_clock::now();
        stats.beginRound(tm.blockType);
        stats.printRoundStatsHeader();
    }
    chrono::duration<float> duration =
        high_resolution_clock::now() - tm.lastT;
    if (duration.count() >= TEST_MODE_STATS_INTERVAL) {
        stats.endRound();
        stats.printRoundStats(duration.count());

        tm.nRounds++;
        if (tm.nRounds >= TEST_MODE_BLOCK_CHANGE_RATE 
            && tm.testCPU 
            && tm.testGPU) {
            auto &bt = tm.blockType;
            bt = (bt == BLOCK_GPU) ? BLOCK_CPU : BLOCK_GPU;
            tm.nRounds = 0;
            cout << endl;
            stats.printRoundStatsHeader();
        }

        tm.lastT = high_resolution_clock::now();
        stats.beginRound(tm.blockType);
    }
}
