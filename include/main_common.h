#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include "../../argon2-gpu/include/commandline/commandlineparser.h"
#include "../../argon2-gpu/include/commandline/argumenthandlers.h"

#include <iomanip>
#include <boost/algorithm/string.hpp>

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

#include "miner_version.h"
#include "perfscope.h"
#include "testMode.h"

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
const uint32_t DEFAULT_BATCH_SIZE = 48;

typedef std::chrono::time_point<std::chrono::high_resolution_clock> t_time_point;

struct OpenCLArguments {
    bool showHelp = false;
    bool listDevices = false;
    bool allDevices = false;
    bool skipCpuBlocks = false;
    bool skipGpuBlocks = false;
    bool testMode = false;
    bool legacyHashrate = false;
    std::vector<uint32_t> deviceIndexList = { 0 };
    std::vector<uint32_t> threadsPerDeviceList = { 1 };
    std::vector<uint32_t> batchSizePerDeviceList = { DEFAULT_BATCH_SIZE };
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
                //std::cout << "--- Device " << it << " is not a GPU, skipping it" << std::endl << devices[it].getName() << std::endl;
                continue;
            }
        }

        size_t deviceIndex = it;
        size_t nThreads = args.threadsPerDeviceList.back();
        if (deviceListItem < args.threadsPerDeviceList.size()) {
            nThreads = args.threadsPerDeviceList[deviceListItem];
        }
        
        cout << "--- Init device " << deviceIndex << ", " << devices[it].getName() << " ---" << endl;
        //cout << devices[it].getInfo() << endl;

        for (int j = 0; j < nThreads; ++j) {
            // choose batch size
            MinerSettings* pNewSettings = new MinerSettings(settings);
            uint32_t batchSize = args.batchSizePerDeviceList.back();
            if (deviceListItem < args.batchSizePerDeviceList.size()) {
                batchSize = args.batchSizePerDeviceList[deviceListItem];
            }

            // create the miner
            Miner *miner = new MINER(stats, pNewSettings, batchSize, updater, &deviceIndex);
            miner->computeCPUBatchSize();
            miners.push_back(miner);
            cout << " - processing unit " << j << ", " << miner->getInfo() << endl;
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
        }, "device(s)", 'd', "GPU devices to use, examples: -d 0 / -d 1 / -d 0,1,3,5", "0", "DEVICES"),

        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.allDevices = true; },
            "use-all-devices", 'u', "use all available GPU devices (overrides -d)"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string address) { state.address = address; }, "address", 'a',
            "public arionum address",
            "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK",
            "ADDRESS"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string poolUrl) { state.poolUrl = poolUrl; }, "pool", 'p',
            "pool URL", "http://aropool.com", "POOL_URL"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string threadsList) {
                if (!parseUInt32List(threadsList, state.threadsPerDeviceList)) {
                    cout << "Error parsing -t parameter, will use -t 1" << endl;
                    state.threadsPerDeviceList = { 1 };
                }
        }, "threads-per-device", 't', "number of threads to use per device, examples: -t 1 / -t 6,3", "1", "THREADS"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string batchSizesList) {
                if (!parseUInt32List(batchSizesList, state.batchSizePerDeviceList)) {
                    cout << "Error parsing -b parameter, will use -b "<< DEFAULT_BATCH_SIZE << endl;
                    state.batchSizePerDeviceList = { DEFAULT_BATCH_SIZE };
                }
        }, "batches-per-device", 'b', "number of batches to use per device, examples: -b 6 / -b 6,3", std::to_string(DEFAULT_BATCH_SIZE), "BATCHES"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string workerId) { state.workerId = workerId; }, "workerId", 'i',
            "worker id", USE_AUTOGENERATED_WORKER_ID, "WORKER_ID"),

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

#ifdef _WIN32
int gettimeofday(struct timeval* p, void* tz) {
    ULARGE_INTEGER ul; // As specified on MSDN.
    FILETIME ft;

    // Returns a 64-bit value representing the number of
    // 100-nanosecond intervals since January 1, 1601 (UTC).
    GetSystemTimeAsFileTime(&ft);

    // Fill ULARGE_INTEGER low and high parts.
    ul.LowPart = ft.dwLowDateTime;
    ul.HighPart = ft.dwHighDateTime;
    // Convert to microseconds.
    ul.QuadPart /= 10ULL;
    // Remove Windows to UNIX Epoch delta.
    ul.QuadPart -= 11644473600000000ULL;
    // Modulo to retrieve the microseconds.
    p->tv_usec = (long)(ul.QuadPart % 1000000LL);
    // Divide to retrieve the seconds.
    p->tv_sec = (long)(ul.QuadPart / 1000000LL);

    return 0;
}
#endif

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

typedef struct MinerStats {
    t_time_point lastT = {};
    BLOCK_TYPE lastTaskType = BLOCK_GPU;
    double lastTaskHashrate = -1.0;
    bool lastTaskValidated = true;
}t_minerStats;

vector<t_minerStats> s_minerStats;

void minerStatsOnNewTask(int minerId, t_time_point time) {
    if (s_minerStats.size() == 0)
        s_minerStats.resize(s_miners.size());

    auto &miner = s_miners[minerId];
    auto &mstats = s_minerStats[minerId];

    if (mstats.lastT == t_time_point()) {
        mstats.lastT = time;
    }
    else {
        std::chrono::duration<double> duration = time - mstats.lastT;
        auto nHashes = miner->getCurrentBatchSize();
        mstats.lastT = time;
        mstats.lastTaskType = miner->getCurrentBlockType();

        if (!mstats.lastTaskValidated) {
            mstats.lastTaskHashrate = 0;
        }
        else {
            mstats.lastTaskHashrate = (double)(nHashes) / duration.count();
        }
    }
}

void minerStatsOnTaskEnd(int minerId, bool hashesAccepted) {
    s_minerStats[minerId].lastTaskValidated = hashesAccepted;
}

double minerStatsGetLastHashrate() {
    double tot = 0.0;
    for (const auto &it : s_minerStats) {
        tot += it.lastTaskHashrate;
    }
    return tot;
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

        if (!miner->mineBlock(blockType)) {
            continue;
        }

        uint32_t batchSize = (blockType == BLOCK_GPU) ? 
            miner->getInitialBatchSize() : 
            miner->getCPUBatchSize();

        PROFILE("reconfigure",
            s_miners[i]->reconfigureArgon(
                Miner::getPasses(blockType),
                Miner::getMemCost(blockType),
                Miner::getLanes(blockType),
                batchSize)
        );

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
                    s_miners[i]->getCurrentBatchSize(),
                    duration.count() * 1000.f,
                    (float)s_miners[i]->getCurrentBatchSize() / duration.count());
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
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
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

    // show version
    cout << getVersionStr() << endl << endl;

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

    if (args.testMode)
        enableTestMode();

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
