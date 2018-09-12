#pragma once

#include "minerdata.h"

class Stats;

bool testMode();
BLOCK_TYPE testModeBlockType();
void updateTestMode(Stats &stats);
int testModeRoundLengthInSeconds();
void enableTestMode();
