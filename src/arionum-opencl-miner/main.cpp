#include "../../argon2-gpu/include/argon2-opencl/globalcontext.h"
#include "../../include/openclminer.h"

#include "../../include/main_common.h"

int main(int, const char *const *argv) {
    return commonMain<opencl::GlobalContext, OpenClMiner>(argv);
}
