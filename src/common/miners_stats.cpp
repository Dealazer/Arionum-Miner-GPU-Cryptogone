#include "../../include/miner.h"
#include "../../include/miners_stats.h"
#include "../../include/minerdata.h"

using std::vector;

typedef struct MinerStats {
    t_time_point lastT = {};
    BLOCK_TYPE lastTaskType = BLOCK_MAX;
    double lastTaskHashrate = -1.0;
    bool lastTaskValidated = false;
}t_minerStats;

vector<t_minerStats> s_minerStats;

void minerStatsOnNewTask(AroMiner & miner, int minerIndex, int nMiners, t_time_point time) {
    if (s_minerStats.size() == 0)
        s_minerStats.resize(nMiners);

    auto &mstats = s_minerStats[minerIndex];

    if (mstats.lastT == t_time_point()) {
        mstats.lastT = time;
    }
    else {
        std::chrono::duration<double> duration = time - mstats.lastT;
        auto nHashes = miner.nHashesPerRun();
        mstats.lastT = time;
        mstats.lastTaskType = miner.providerBlockType();

        if (!mstats.lastTaskValidated) {
            mstats.lastTaskHashrate = 0;
        }
        else {
            mstats.lastTaskHashrate = (double)(nHashes) / duration.count();
        }
    }
}

void minerStatsOnTaskEnd(int minerId, bool hashesAccepted) {
    s_minerStats[minerId].lastTaskValidated = hashesAccepted;
}

double minerStatsGetLastHashrate(BLOCK_TYPE b) {
    double tot = 0.0;
    for (const auto &it : s_minerStats) {
        if (it.lastTaskType == b)
            tot += it.lastTaskHashrate;
    }
    return tot;
}