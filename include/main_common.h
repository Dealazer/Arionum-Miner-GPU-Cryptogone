#include "miner_version.h"
#include "mining_system.h"
#include "test_services.h"
#include "pool_services.h"

#include "../../argon2-gpu/include/commandline/commandlineparser.h"
#include "../../argon2-gpu/include/commandline/argumenthandlers.h"

#ifdef _WIN32
#include <windows.h>
#include "../../include/win_tools.h"
#else
#define sscanf_s sscanf

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

void handler(int sig) {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}
#endif

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <iomanip>
#include <boost/algorithm/string.hpp>

// cpprest lib
#pragma comment(lib, "cpprest_2_10")

// openSSL libs
#pragma comment(lib, "ssleay32")
#pragma comment(lib, "libeay32")

using namespace argon2;
using namespace std;
using namespace libcommandline;

const string USE_AUTOGENERATED_WORKER_ID = "autogenerated";
const uint32_t DEFAULT_GPU_BATCH_SIZE = 0; // 0 <-> auto
const uint32_t DEFAULT_TASKS_PER_DEVICE = 1;

struct OpenCLArguments {
    bool showHelp = false;
    bool listDevices = false;
    bool allDevices = false;
    bool skipCpuBlocks = false;
    bool skipGpuBlocks = false;
    bool testMode = false;
    bool legacyHashrate = false;
    std::vector<uint32_t> deviceIndexList = { 0 };
    std::vector<uint32_t> nTasksPerDeviceList = { DEFAULT_TASKS_PER_DEVICE };
    std::vector<uint32_t> gpuBatchSizePerDeviceList = { DEFAULT_GPU_BATCH_SIZE };
    string workerId = USE_AUTOGENERATED_WORKER_ID;
    string address = "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK";
    string poolUrl = "http://aropool.com";
    argon2::OPT_MODE cpuBlocksOptimizationMode = PRECOMPUTE_LOCAL_STATE;
};

extern bool s_miningReady;

bool parseUInt32List(const std::string &s, std::vector<uint32_t> &ol) {
    std::vector<string> ls;
    boost::split(ls, s, boost::is_any_of(","));
    ol.clear();
    for (auto& it : ls) {
        uint32_t id;
        auto idStr = boost::trim_left_copy(it);
        auto count = sscanf_s(idStr.c_str(), "%u", &id);
        if (count == 1) {
            ol.push_back(id);
        }
        else {
            return false;
        }
    }
    return true;
}

CommandLineParser<OpenCLArguments> buildCmdLineParser() {
    static const auto positional = PositionalArgumentHandler<OpenCLArguments>(
        [](OpenCLArguments &, const std::string &) {});

    std::vector<const CommandLineOption<OpenCLArguments> *> options{
        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.listDevices = true; },
            "list-devices", 'l', "list all available devices and exit"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string deviceList) {
                if (!parseUInt32List(deviceList, state.deviceIndexList)) {
                    cout << "Error parsing -d parameter, will use -d 0" << endl;
                    state.deviceIndexList = { 0 };
                }
        }, "device(s)", 'd', "GPU devices to use, examples: -d 0 / -d 1 / -d 0,1,3,5", "0", "devices"),

        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.allDevices = true; },
            "use-all-devices", 'u', "use all available GPU devices (overrides -d)"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string address) { state.address = address; }, "address", 'a',
            "public arionum address",
            "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK",
            "address"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string poolUrl) { state.poolUrl = poolUrl; }, "pool", 'p',
            "pool URL", "http://aropool.com", "pool_utl"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string nTasksList) {
                if (!parseUInt32List(nTasksList, state.nTasksPerDeviceList)) {
                    cout << "Error parsing -t parameter, will use default" << endl;
                    state.nTasksPerDeviceList = { DEFAULT_TASKS_PER_DEVICE };
                }
        }, "tasks-per-device", 't', "number of parallel tasks per device, examples: -t 1 / -t 6,3", 
            std::to_string(DEFAULT_TASKS_PER_DEVICE), "nTaskPerDevice"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string batchSizeList) {
                if (!parseUInt32List(batchSizeList, state.gpuBatchSizePerDeviceList)) {
                    cout << "Error parsing -b parameter, will use default" << endl;
                    state.gpuBatchSizePerDeviceList = { DEFAULT_GPU_BATCH_SIZE };
                }
        }, "gpu-batch-size", 'b', "GPU batch size, examples: -b 224 / -b 224,196", 
            std::to_string(DEFAULT_GPU_BATCH_SIZE), "batchSize"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string workerId) { state.workerId = workerId; }, "workerId", 'i',
            "worker id", USE_AUTOGENERATED_WORKER_ID, "worker_id"),

        new FlagOption<OpenCLArguments>(
                [](OpenCLArguments &state) { state.cpuBlocksOptimizationMode = PRECOMPUTE_SHUFFLE; },
                "use-shuffle-kernel-cpu", '\0', "use index_shuffle kernel for cpu blocks, may be faster on some systems"),

        new FlagOption<OpenCLArguments>(
                [](OpenCLArguments &state) { state.skipCpuBlocks = true; },
                "skip-cpu-blocks", '\0', "do not mine cpu blocks"),
 
        new FlagOption<OpenCLArguments>(
                [](OpenCLArguments &state) { state.skipGpuBlocks = true; },
                "skip-gpu-blocks", '\0', "do not mine gpu blocks"),

        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.testMode = true; },
            "test-mode", '\0', "test CPU/GPU blocks hashrate"),

        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.legacyHashrate = true; },
            "legacy-5s-hashrate", '\0', "show 5s avg hashrate instead of last set of batches hashrate"),

        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.showHelp = true; },
                "help", '?', "show this help and exit")
    };

    return CommandLineParser<OpenCLArguments>(
        "",
        positional, options);
}

template <class CONTEXT>
void printDeviceList() {
    CONTEXT global;
    auto &devices = global.getAllDevices();
    if (!devices.size()) {
        cout << "No device found !" << endl;
        return;
    }

    for (size_t i = 0; i < devices.size(); i++) {
        auto &device = devices[i];
        cout << "Device #" << i << ": " << device.getName()
            << endl;
    }
}

string generateUniqid() {
    struct timeval tv {};
    gettimeofday(&tv, nullptr);
    auto sec = (int)tv.tv_sec;
    auto usec = (int)(tv.tv_usec % 0x100000);
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(8) << std::hex << sec << std::setfill('0') << std::setw(5) << std::hex
        << usec;
    return ss.str();
}



template <class CONTEXT, class DEVICE, class MINER>
int run(const char *const *argv) {
    using MiningSystem_t = MiningSystem<CONTEXT, DEVICE, MINER>;

#ifdef _MSC_VER
    // set a fixed console size (default is not wide enough)
    setConsoleSize(150, 40, 2000);
#else
    // install error handler to show stack trace
    signal(SIGSEGV, handler);
#endif

    // show version & extra info
    cout << getVersionStr();
    auto extra = getExtraInfoStr();
    if (extra.size() > 0)
        cout << ", " << extra;
    cout << endl << endl;

    // process arguments
    CommandLineParser<OpenCLArguments> parser = buildCmdLineParser();
    OpenCLArguments args;
    int ret = parser.parseArguments(args, argv);
    if (ret != 0) {
        return 1;
    }

    if (args.showHelp) {
        parser.printHelp(argv);
        return 0;
    }

    if (args.testMode) {
        if (args.skipCpuBlocks && args.skipGpuBlocks) {
            std::cout << "Cannot skip both CPU & GPU blocks in test mode, aborting."<< std::endl;
            exit(1);
        }
        enableTestMode(!args.skipCpuBlocks, !args.skipGpuBlocks);
    }

    string uniqid = (args.workerId == USE_AUTOGENERATED_WORKER_ID) ?
        generateUniqid() :
        args.workerId;

    if (args.allDevices) {
        args.deviceIndexList.clear();

        std::unique_ptr<CONTEXT> ctx(new CONTEXT());
        auto &devices = ctx->getAllDevices();
        for (int i = 0; i < devices.size(); ++i)
            args.deviceIndexList.push_back(i);
    }

    // check how many devices we have
    {
        std::cout << "Initializing " << API_NAME << std::endl;
        CONTEXT global;
        auto &devices = global.getAllDevices();
        if (devices.size() == 0) {
            std::cout << "No mining device found, aborting..." << std::endl;
            return 1;
        }
        std::cout << "Found " << devices.size() << " compute devices" << std::endl;
    }

    // list devices mode
    if (args.listDevices) {
        printDeviceList<CONTEXT>();
        return 0;
    }

    // devices configurations
    auto devicesConfigs = [&] () -> auto {
        auto itemOrLast = [](int index, std::vector<uint32_t> v) -> uint32_t {
            if (index < v.size())
                return v[index];
            return v[v.size() - 1];
        };

        std::vector<MiningSystem_t::DeviceConfig> configs;
        for (auto deviceIndex : args.deviceIndexList)
            configs.push_back({
            deviceIndex,
            itemOrLast(deviceIndex, args.nTasksPerDeviceList),
            itemOrLast(deviceIndex, args.gpuBatchSizePerDeviceList),
            args.cpuBlocksOptimizationMode
        });
        return configs;
    }();

    // MinerSettings
    size_t dummy = 0;
    MinerSettings minerSettings(
        &args.poolUrl, &args.address, &uniqid, &dummy,
        !args.skipGpuBlocks, !args.skipCpuBlocks, !args.legacyHashrate);
    std::cout << minerSettings << std::endl;

    // create stats module
    Stats stats;

    // create mining system
    std::unique_ptr<Updater> updater;
    std::unique_ptr<thread> updateThread;

    auto miningSystem = [&]() -> auto {
        if (testMode()) {
            return std::unique_ptr<MiningSystem_t>(new MiningSystem_t(
                devicesConfigs,
                std::unique_ptr<AroNonceProviderTestMode>(new AroNonceProviderTestMode(stats)),
                std::unique_ptr<AroResultsProcessorTestMode>(new AroResultsProcessorTestMode()),
                stats));
        }

        updater.reset(new Updater(stats, minerSettings));
        updateThread.reset(new std::thread(&Updater::start, updater.get()));
        updateThread->detach();
        return std::unique_ptr<MiningSystem_t>(new MiningSystem_t(
            devicesConfigs,
            std::unique_ptr<AroNonceProviderPool>(new AroNonceProviderPool(*updater)),
            std::unique_ptr<AroResultsProcessorPool>(new AroResultsProcessorPool(minerSettings, stats)), 
            stats));
    }();

    // create miners
    miningSystem->createMiners(devicesConfigs);
    s_miningReady = true;

    // mine
    miningSystem->miningLoop();

    return true;
}

template <class CONTEXT, class DEVICE, class MINER>
int commonMain(const char *const *argv) {
    try {        
        return run<CONTEXT, DEVICE, MINER>(argv);
    }
    catch (std::logic_error e) {
        std::cout << "Exception in main thread: " << e.what() << std::endl;
        return 1;
    }
    catch (exception e) {
        std::cout << "Exception in main thread: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
