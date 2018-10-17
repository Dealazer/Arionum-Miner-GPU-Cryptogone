#pragma once

typedef std::chrono::time_point<std::chrono::high_resolution_clock> t_time_point;

void minerStatsOnNewTask(int minerId, t_time_point time);
void minerStatsOnTaskEnd(int minerId, bool hashesAccepted);
double minerStatsGetLastHashrate(BLOCK_TYPE);