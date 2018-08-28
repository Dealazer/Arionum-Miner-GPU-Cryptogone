#pragma once

#ifdef PROFILE
#include <chrono>
using std::chrono::high_resolution_clock;

class PerfScope {
public:
    PerfScope(const std::string &_comment, bool _always_dump = false) : comment(_comment), always_dump(_always_dump) {
        startT = high_resolution_clock::now();
    }

    ~PerfScope() {
        std::chrono::duration<float> duration = high_resolution_clock::now() - startT;
        float durationMs = duration.count() * 1000.f;
        if (always_dump) {
            std::cout << "|" << comment << "| => " << std::fixed << std::setprecision(2) << durationMs << " ms" << std::endl;
        }
        else if (durationMs >= 1.0f) {
            std::cout << "LONG |" << comment << "| => " << std::fixed << std::setprecision(2) << durationMs << " ms" << std::endl;
        }
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> startT;
    std::string comment;
    bool always_dump;
};
#else
class PerfScope {
public:
    PerfScope(const std::string &_comment, bool _always_dump = false) {};
};
#endif