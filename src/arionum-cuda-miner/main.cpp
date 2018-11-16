//
// Created by guli on 01/02/18.
//
#include "../../include/cudaminer.h"
#include <sstream>

#define API_NAME "CUDA"
const std::string DEFAULT_CPU_KERNEL = "shuffle";

std::string getExtraInfoStr() {
    std::ostringstream oss;
    oss << "CUDART_VERSION=";
#ifndef CUDART_VERSION
    oss << "unknown";
#else    
    oss << CUDART_VERSION;
#endif
    return oss.str();
}

#include "../../include/main_common.h"

int main(int, const char *const *argv) {
    return commonMain<
        cuda::GlobalContext, CudaMiningDevice, CudaMiner>(argv);
}
