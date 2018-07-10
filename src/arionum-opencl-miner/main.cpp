#include "../../argon2-gpu/include/argon2-opencl/globalcontext.h"
#include "../../include/openclminer.h"

#define OPENCL_MINER (1)
#include "../../include/main_common.h"

int main(int, const char *const *argv) {
    return commonMain<opencl::GlobalContext, OpenClMiner>(argv);
}
