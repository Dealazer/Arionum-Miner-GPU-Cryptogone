#include "../../argon2-gpu/include/argon2-opencl/globalcontext.h"
#include "../../include/openclminer.h"

#define API_NAME "OPENCL"
const std::string DEFAULT_CPU_KERNEL = "local_state";

std::string getExtraInfoStr() {
    std::ostringstream oss;
    oss << "OpenCL ";
#if CL_VERSION_2_2
    oss << "2.2";
#elif CL_VERSION_2_1
    oss << "2.1";
#elif CL_VERSION_1_2
    oss << "1.2";
#elif CL_VERSION_1_1
    oss << "1.1";
#elif CL_VERSION_1_0
    oss << "1.0";
#endif
    return oss.str();
}

#include "../../include/main_common.h"

int main(int, const char *const *argv) {
    return commonMain<
        opencl::GlobalContext, OpenClMiningDevice, OpenClMiner>(argv);
}
