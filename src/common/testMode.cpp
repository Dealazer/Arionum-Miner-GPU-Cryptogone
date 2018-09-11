#include <iostream>
#include <atomic>
#include <chrono>
#include <mutex>
#include <climits>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "../include/testMode.h"
#include "../include/stats.h"

using namespace std;
using std::chrono::high_resolution_clock;

typedef std::chrono::time_point<high_resolution_clock> t_time_point;

const int TEST_MODE_STATS_INTERVAL = 2;
const int TEST_MODE_BLOCK_CHANGE_RATE = 5;

struct TestModeInfo {
    bool enabled = true;
    t_time_point lastT = {};
    int nRounds = 0;
    BLOCK_TYPE blockType = BLOCK_GPU;
};

static TestModeInfo s_testMode;

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
    if (!s_testMode.enabled)
        return;
    if (s_testMode.lastT == t_time_point()) {
        s_testMode.lastT = high_resolution_clock::now();
        stats.beginRound(s_testMode.blockType);
        cout << "--- Test Mode starts ---" << endl << endl;
        stats.printRoundStatsHeader();
    }
    chrono::duration<float> duration =
        high_resolution_clock::now() - s_testMode.lastT;
    if (duration.count() >= TEST_MODE_STATS_INTERVAL) {
        stats.endRound();
        stats.printRoundStats(duration.count());

        s_testMode.nRounds++;
        if (s_testMode.nRounds >= TEST_MODE_BLOCK_CHANGE_RATE) {
            auto &bt = s_testMode.blockType;
            bt = (bt == BLOCK_GPU) ? BLOCK_CPU : BLOCK_GPU;
            s_testMode.nRounds = 0;
            cout << endl;
            stats.printRoundStatsHeader();
        }

        s_testMode.lastT = high_resolution_clock::now();
        stats.beginRound(s_testMode.blockType);
    }
}
