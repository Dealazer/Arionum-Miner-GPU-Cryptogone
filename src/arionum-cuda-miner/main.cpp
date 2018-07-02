//
// Created by guli on 01/02/18.
//
#include "../../include/cudaminer.h"
#include "../../include/main_common.h"

int main(int, const char *const *argv) {
    return commonMain<cuda::GlobalContext, CudaMiner>(argv);
}
