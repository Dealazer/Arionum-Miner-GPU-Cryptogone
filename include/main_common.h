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

#include "../../include/miner_version.h"

//#define PROFILE
#include "../../include/perfscope.h"

// cpprest lib
#pragma comment(lib, "cpprest_2_10")

// openSSL libs
#pragma comment(lib, "ssleay32")
#pragma comment(lib, "libeay32")

#define STR_IMPL_(x) #x

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
        }, "batches-per-device", 'b', "number of batches to use per device, examples: -b 6 / -b 6,3", STR_IMPL_(DEFAULT_BATCH_SIZE), "BATCHES"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string workerId) { state.workerId = workerId; }, "workerId", 'i',
            "worker id", USE_AUTOGENERATED_WORKER_ID, "WORKER_ID"),

        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.showHelp = true; },
                "help", '?', "show this help and exit")
    };

    return CommandLineParser<OpenCLArguments>(
        "A tool for testing the argon2-opencl and argon2-cuda libraries.",
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

bool feedMiners() {
    for (int i = 0; i < s_miners.size(); i++) {
        if (s_minerIdle[i] == false)
            continue;

        auto &miner = s_miners[i];
        auto data = s_pUpdater->getData();
        if (data.isValid()==false)
            return false;
        
        auto blockType = (TEST_MODE == TEST_CPU) ? (BLOCK_CPU) : 
                            ((TEST_MODE == TEST_GPU) ? BLOCK_GPU : 
                            data.getBlockType());
        
        if (blockType != BLOCK_CPU && blockType != BLOCK_GPU) {
            cout << ("ERROR: unknown block type !!!\n") << endl;
            exit(1);
        }

#define DEBUG_DURATIONS (0)
#if DEBUG_DURATIONS
        if (s_minerStartT[i] != t_time_point()) {
            std::chrono::duration<float> T = high_resolution_clock::now() - s_start;
            std::chrono::duration<float> duration = high_resolution_clock::now() - s_minerStartT[i];
            printf("T=%4.2f miner %d, %d batches in %.2fms\n",
                T.count(),
                i,
                s_miners[i]->getCurrentBatchSize(),
                duration.count() * 1000.f);
        }
        s_minerStartT[i] = high_resolution_clock::now();
#endif

        uint32_t batchSize = 
            (blockType == BLOCK_GPU) ? miner->getInitialBatchSize() : miner->getCPUBatchSize();

        s_miners[i]->reconfigureArgon(
            Miner::getPasses(blockType),
            Miner::getMemCost(blockType),
            Miner::getLanes(blockType),
            batchSize);

        miner->hostPrepareTaskData();
        miner->deviceUploadTaskDataAsync();
        miner->deviceLaunchTaskAsync();
        miner->deviceFetchTaskResultAsync();

#if 0
        std::chrono::duration<float> duration = high_resolution_clock::now() - s_minerStartT[i];
        printf("miner %d, new task set in %.2fms\n",
            i, duration.count() * 1000.f);
#endif

        s_minerIdle[i] = false;
    }
    return true;
}

int miningLoop() {
    while (true) {
        // give new work to idle miners
        bool ok = feedMiners();
        if (!ok) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // process miner results
        int nIdle = 0;
        for (int i = 0; i < s_miners.size(); i++) {
            if (s_minerIdle[i]) {
                nIdle++;
                continue;
            }
            if (s_miners[i]->deviceResultsReady()) {
                s_miners[i]->hostProcessResults();
                s_minerIdle[i] = true;
            }
        }

        // wait a bit, to avoid polling too much GPUs about tasks status
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
#if (TEST_MODE == TEST_CPU)
    cout << endl << "-------------- TEST CPU ----------------" << endl << endl;
#elif TEST_MODE == TEST_GPU
    cout << endl << "-------------- TEST GPU ----------------" << endl << endl;
#endif

    // basic check to see if CUDA / OpenCL is supported
    std::cout << "Initializing " << API_NAME << std::endl;
    CONTEXT global;
    auto &devices = global.getAllDevices();
    std::cout << "Found " << devices.size() << " compute devices"<< std::endl;

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

    if (args.listDevices) {
        printDeviceList<CONTEXT>();
        return 0;
    }

    size_t dummy = 0;
    string uniqid = (args.workerId == USE_AUTOGENERATED_WORKER_ID) ?
                        generateUniqid() :
                        args.workerId;
    MinerSettings settings(&args.poolUrl, &args.address, &uniqid, &dummy);
    std::cout << settings << std::endl;

    auto *stats = new Stats();


    s_pUpdater = new Updater(stats, &settings);
    thread updateThread(&Updater::start, s_pUpdater);
    updateThread.detach();

    spawnMiners<CONTEXT, MINER>(args, s_miners, stats, s_pUpdater, settings);
    s_miningReady = true;

    miningLoop();

    return true;
}

template <class CONTEXT, class MINER>
int commonMain(const char *const *argv) {
    try {        
        return run<CONTEXT, MINER>(argv);
    }
    catch (exception e) {
        std::cout << "Exception in main thread: " << e.what() << std::endl;
        exit(1);
    }
    return 1;
}
