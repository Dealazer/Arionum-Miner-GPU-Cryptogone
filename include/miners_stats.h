#pragma once

#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> t_time_point;

class AroMiner;
void minerStatsOnNewTask(AroMiner & miner, int minerIndex, int nMiners, t_time_point time);
void minerStatsOnTaskEnd(int minerId, bool hashesAccepted);
double minerStatsGetLastHashrate(BLOCK_TYPE);