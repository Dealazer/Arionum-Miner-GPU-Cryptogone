#include "miner_version.h"
#include "perfscope.h"
#include "testMode.h"
#include "miners_stats.h"

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include <iomanip>
#include <boost/algorithm/string.hpp>

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

// cpprest lib
#pragma comment(lib, "cpprest_2_10")

// openSSL libs
#pragma comment(lib, "ssleay32")
#pragma comment(lib, "libeay32")

using namespace argon2;
using namespace std;
using namespace libcommandline;
using std::chrono::high_resolution_clock;

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
};

extern bool s_miningReady;

Updater* s_pUpdater = nullptr;
vector<Miner *> s_miners;
vector<bool> s_minerIdle;

vector<t_time_point> s_minerStartT(256);
t_time_point s_start = high_resolution_clock::now();

template <class CONTEXT, class MINER>
void spawnMiners(OpenCLArguments &args, vector<Miner *> &miners, Stats* stats, Updater *updater, MinerSettings& settings) {
    
    auto global = new CONTEXT();
    auto &devices = global->getAllDevices();

    // -u option
    if (args.allDevices) {
        args.deviceIndexList.clear();
        for (int i = 0; i < devices.size(); ++i) {
            args.deviceIndexList.push_back(i);
        }
    }

    //
    if (args.deviceIndexList.size() == 0) {
        std::cout << "Error: no device found, aborting" << std::endl;
        exit(1);
    }

    // create miners
    cout << endl;

    int deviceListItem = -1;
    for (const auto &it : args.deviceIndexList) {
        deviceListItem++;
        if (it >= devices.size()) {
            cout << endl;
            cout << "--- Device " << it << " does not exist, skipping it" << endl;
            cout << endl;
            continue;
        }

        // skip CPU devices (only needed for OpenCL)
        if (!strcmp(API_NAME, "OPENCL")) {
            auto devInfo = devices[it].getInfo();
            if (devInfo.find("Type: GPU") == std::string::npos) {
                //std::cout << std::endl;
                //std::cout 
                //  << "--- Device " << it << " is not a GPU, skipping it" 
                //  << std::endl << devices[it].getName() << std::endl;
                continue;
            }
        }

        uint32_t deviceIndex = it;
        uint32_t nThreads = args.nTasksPerDeviceList.back();
        if (deviceListItem < args.nTasksPerDeviceList.size()) {
            nThreads = args.nTasksPerDeviceList[deviceListItem];
        }

        uint32_t nGPUBatches = args.gpuBatchSizePerDeviceList.back();
        if (deviceListItem < args.gpuBatchSizePerDeviceList.size()) {
            nGPUBatches = args.gpuBatchSizePerDeviceList[deviceListItem];
        }

        cout << "--- Device " << deviceIndex << ", " << devices[it].getName() << " ---" << endl;
        //cout << devices[it].getInfo() << endl;

        auto pMiningDevice = new OpenClMiningDevice(
            deviceIndex, nThreads, nGPUBatches);

        for (uint32_t j = 0; j < nThreads; ++j) {
            cout << " - create miner " << j << endl;
            Miner *miner = new MINER(
                pMiningDevice->getProgramContext(),
                pMiningDevice->getQueue(j),
                pMiningDevice->getMemConfig(j),
                stats, settings, updater);
            miners.push_back(miner);
        }
        cout << endl;
    }

    s_minerIdle = vector<bool>(s_miners.size(), true);
}

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

//bool parseFloatList(const std::string &s, std::vector<double> &ol) {
//    std::vector<string> ls;
//    boost::split(ls, s, boost::is_any_of(","));
//    ol.clear();
//    for (auto& it : ls) {
//        double v;
//        auto idStr = boost::trim_left_copy(it);
//        auto count = sscanf_s(idStr.c_str(), "%lf", &v);
//        if (count == 1) {
//            ol.push_back(v);
//        }
//        else {
//            return false;
//        }
//    }
//    return true;
//}

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

bool feedMiners(Stats *stats) {
    for (int i = 0; i < s_miners.size(); i++) {
        
        t_time_point now = high_resolution_clock::now();

        if (s_minerIdle[i] == false)
            continue;

        auto &miner = s_miners[i];

        static bool s_miningStartedMsgShown = false;
        if (!s_miningStartedMsgShown) {
            stats->printTimePrefix();
            cout << 
                (testMode() ? 
                    "--- Start Testing ---" : "--- Start Mining ---")
                 << endl;
            s_miningStartedMsgShown = true;
        }

        minerStatsOnNewTask(i, now);

        updateTestMode(*stats);

        BLOCK_TYPE blockType;
        
        if (testMode()) {
            blockType = testModeBlockType();
        }
        else {
            auto data = s_pUpdater->getData();
            if (data.isValid() == false)
                return false;
            blockType = data.getBlockType();
        }

        if (!miner->canMineBlock(blockType)) {
            continue;
        }

        PROFILE("reconfigure",
            s_miners[i]->reconfigureArgon(
                Miner::getPasses(blockType),
                Miner::getMemCost(blockType),
                Miner::getLanes(blockType)));
        PROFILE("prepareHashes", miner->hostPrepareTaskData());
        PROFILE("uploadHashes", miner->deviceUploadTaskDataAsync());
        PROFILE("runKernel", miner->deviceLaunchTaskAsync());
        PROFILE("fetchResults", miner->deviceFetchTaskResultAsync());

#if 0
        std::chrono::duration<float> duration = high_resolution_clock::now() - s_minerStartT[i];
        printf("miner %d, new task set in %.2fms\n",
            i, duration.count() * 1000.f);
#endif

        s_minerIdle[i] = false;
    }
    return true;
}

int processMinersResults() {
    int nIdle = 0;
    for (int i = 0; i < s_miners.size(); i++) {
        if (s_minerIdle[i]) {
            nIdle++;
            continue;
        }
        if (s_miners[i]->deviceResultsReady()) {
#define DEBUG_DURATIONS (0)
#if DEBUG_DURATIONS
            if (s_minerStartT[i] != t_time_point()) {
                std::chrono::duration<float> T = high_resolution_clock::now() - s_start;
                std::chrono::duration<float> duration = high_resolution_clock::now() - s_minerStartT[i];
                printf("T=%4.3f miner %d, %d hashes in %.2fms => %.1f Hs\n",
                    T.count(),
                    i,
                    s_miners[i]->getNbHashesPerIteration(),
                    duration.count() * 1000.f,
                    (float)s_miners[i]->getNbHashesPerIteration() / duration.count());
            }
            s_minerStartT[i] = high_resolution_clock::now();
#endif
            bool hashesAccepted = s_miners[i]->hostProcessResults();
            s_minerIdle[i] = true;

            minerStatsOnTaskEnd(i, hashesAccepted);
        }
    }
    return nIdle;
}

int miningLoop(Stats *stats) {
    while (true) {
        bool ok = feedMiners(stats);
        if (!ok) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        int nIdle = processMinersResults();
        if (nIdle == 0) {
            //std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

template <class CONTEXT, class MINER>
int run(const char *const *argv) {
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

    // basic check to see if CUDA / OpenCL is supported
    std::cout << "Initializing " << API_NAME << std::endl;
    CONTEXT global;
    auto &devices = global.getAllDevices();
    std::cout << "Found " << devices.size() << " compute devices" << std::endl;

    if (args.listDevices) {
        printDeviceList<CONTEXT>();
        return 0;
    }

    size_t dummy = 0;
    string uniqid = (args.workerId == USE_AUTOGENERATED_WORKER_ID) ?
                        generateUniqid() :
                        args.workerId;
    MinerSettings settings(
        &args.poolUrl, &args.address, &uniqid, &dummy,
        !args.skipGpuBlocks, !args.skipCpuBlocks, !args.legacyHashrate);
    std::cout << settings << std::endl;

    auto *stats = new Stats(&settings);

    if (!testMode()) {
        s_pUpdater = new Updater(stats, &settings);
        thread updateThread(&Updater::start, s_pUpdater);
        updateThread.detach();
    }

    spawnMiners<CONTEXT, MINER>(args, s_miners, stats, s_pUpdater, settings);

#if 0
    // warmup
    cout << "Warming up..." << endl;
    for (int i = 0; i < s_miners.size(); i++) {
        auto &miner = s_miners[i];
        miner->hostPrepareTaskData();
        miner->deviceUploadTaskDataAsync();
        miner->deviceLaunchTaskAsync();
        miner->deviceFetchTaskResultAsync();
    }
    for (int i = 0; i < s_miners.size(); i++) {
        auto &miner = s_miners[i];
        miner->deviceWaitForResults();
    }
    // warmup
#endif

    s_miningReady = true;

    miningLoop(stats);

    return true;
}

template <class CONTEXT, class MINER>
int commonMain(const char *const *argv) {
    try {        
        return run<CONTEXT, MINER>(argv);
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
