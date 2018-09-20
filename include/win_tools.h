#pragma once

void setConsoleSize(int width, int height, int bufferHeight);

typedef void(*ctrlCFnPtr_t)(void);
bool setCtrlCHandler(ctrlCFnPtr_t pFn);

#ifdef _WIN32
int gettimeofday(struct timeval* p, void* tz);
#endif