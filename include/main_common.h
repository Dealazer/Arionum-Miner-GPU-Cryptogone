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
#endif

#include "../../include/miner_version.h"

// cpprest lib
#pragma comment(lib, "cpprest_2_10")

// openSSL libs
#pragma comment(lib, "ssleay32")
#pragma comment(lib, "libeay32")

using namespace argon2;
using namespace std;
using namespace libcommandline;

struct OpenCLArguments {
    bool showHelp = false;
    bool listDevices = false;
    bool allDevices = false;
    std::vector<size_t> deviceIndexList = { 0 };
    std::vector<size_t> threadsPerDeviceList = { 1 };
    std::vector<size_t> batchSizePerDeviceList = { 1 };
    string address = "419qwxjJLBRdAVttxcJVT84vdmdq3GP6ghXdQdxN6sqbdr5SBXvPU8bvfVzUXWrjrNrJbAJCvW9JYDWvxenus1pK";
    string poolUrl = "http://aropool.com";
    double d = 1;
};

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
    int deviceListItem = -1;
    for (const auto &it : args.deviceIndexList) {
        deviceListItem++;
        if (it >= devices.size()) {
            std::cout << std::endl;
            std::cout << "--- Device " << it << " does not exist, skipping it" << std::endl;
            continue;
        }
#ifdef OPENCL_MINER
        auto devInfo = devices[it].getInfo();
        if (devInfo.find("Type: GPU") == std::string::npos) {
            std::cout << std::endl;
            std::cout << "--- Device " << it << " is not a GPU, skipping it" << std::endl << devices[it].getName() << std::endl;
            continue;
        }
#endif
        size_t deviceIndex = it;
        size_t nThreads = args.threadsPerDeviceList.back();
        if (deviceListItem < args.threadsPerDeviceList.size()) {
            nThreads = args.threadsPerDeviceList[deviceListItem];
        }

        for (int j = 0; j < nThreads; ++j) {
            // choose batch size
            MinerSettings* pNewSettings = new MinerSettings(settings);
            pNewSettings->batchSize = &args.batchSizePerDeviceList.back();
            if (deviceListItem < args.batchSizePerDeviceList.size()) {
                pNewSettings->batchSize = &args.batchSizePerDeviceList[deviceListItem];
            }

            // create the miner
            std::cout << std::endl;
            cout << "--- Device " << deviceIndex << ", Thread " << j << endl;
            Miner *miner = new MINER(stats, pNewSettings, updater, &deviceIndex);
            miners.push_back(miner);
            miner->printInfo();
        }
    }
}

bool parseIntList(const std::string &s, std::vector<size_t> &ol) {
    std::vector<string> ls;
    boost::split(ls, s, boost::is_any_of(","));
    ol.clear();
    for (auto& it : ls) {
        size_t id;
        auto idStr = boost::trim_left_copy(it);
        auto count = sscanf_s(idStr.c_str(), "%llu", &id);
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
                if (!parseIntList(deviceList, state.deviceIndexList)) {
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
            makeNumericHandler<OpenCLArguments, double>([](OpenCLArguments &state, double devFee) {
                state.d = devFee <= 0.5 ? 1 : devFee;
            }), "dev-donation", 'D', "developer donation", "1", "PERCENTAGE"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string threadsList) {
                if (!parseIntList(threadsList, state.threadsPerDeviceList)) {
                    cout << "Error parsing -t parameter, will use -t 1" << endl;
                    state.threadsPerDeviceList = { 1 };
                }
        }, "threads-per-device", 't', "number of threads to use per device, examples: -t 1 / -t 6,3", "1", "THREADS"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string batchSizesList) {
                if (!parseIntList(batchSizesList, state.batchSizePerDeviceList)) {
                    cout << "Error parsing -b parameter, will use -b 1" << endl;
                    state.batchSizePerDeviceList = { 1 };
                }
        }, "batch-per-device", 'b', "number of batches to use per device, examples: -b 6 / -b 6,3", "1", "BATCHES"),

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

extern bool gMiningStarted;

Updater* s_pUpdater = nullptr;

template <class CONTEXT, class MINER>
int commonMain(const char *const *argv) {
#ifdef _MSC_VER
    // set a fixed console size (default is not wide enough)
    setConsoleSize(150, 40, 2000);
#endif

    // show version
    cout << getVersionStr() << endl << endl;

#ifdef TEST_GPU_BLOCK
    cout << endl << "-------------- TEST MODE ----------------" << endl << endl;
#endif

    // basic check to see if CUDA / OpenCL is supported
    try {
#ifdef OPENCL_MINER
        std::cout << "Initializing OpenCL";
#else
        std::cout << "Initializing CUDA";
#endif
        std::cout << " (if it crashes here, try install latest GPU drivers)" << std::endl;
        CONTEXT global;
        auto &devices = global.getAllDevices();
    }
    catch (exception e) {
        std::cout << "Exception: " << e.what() << std::endl;
        exit(1);
    }

    // parse CLI args
    CommandLineParser<OpenCLArguments> parser = buildCmdLineParser();
    OpenCLArguments args;
    int ret = parser.parseArguments(args, argv);

    // param errors / help / list
    if (ret != 0) {
        return false;
    }
    if (args.showHelp) {
        parser.printHelp(argv);
        return false;
    }
    if (args.listDevices) {
        printDeviceList<CONTEXT>();
        return false;
    }

    // initialize settings
    size_t dummy = 0;
    string uniqid = generateUniqid();
    MinerSettings settings(&args.poolUrl, &args.address, &uniqid, &dummy);
    auto *stats = new Stats(args.d);

    // show launch settings in console
    std::cout << settings << std::endl;

    // launch updater thread
    auto *updater = new Updater(stats, &settings);
    s_pUpdater = updater;
    updater->update();
    thread t(&Updater::start, updater);

    // chooses GPU devices to run on & create miners
    vector<Miner *> miners;
    spawnMiners<CONTEXT, MINER>(args, miners, stats, updater, settings);

    // notify updater that mining started
    gMiningStarted = true;

    // ---- main loop ALTERNATE
    vector<bool> minerIdle(miners.size(), true);
    while (true) {
        //
        bool cpuBlock = false;

        // find idle miners and launch async GPU tasks for them
        for (int i = 0; i < miners.size(); i++) {
            if (minerIdle[i] == true) {
                auto data = updater->getData();
#ifdef TEST_GPU_BLOCK
                if (true) {
#else
                if (data.getType() == BLOCK_GPU) {
#endif
                    minerIdle[i] = false;
                    miners[i]->hostPrepareTaskData();
                    miners[i]->deviceUploadTaskDataAsync();
                    miners[i]->deviceLaunchTaskAsync();
                    miners[i]->deviceFetchTaskResultAsync();
                }
                else {
                    cpuBlock = true;
                }
            }
        }

        // find miners which have finished work, process results & mark them idle
        int nIdle = 0;
        for (int i = 0; i < miners.size(); i++) {
            if (minerIdle[i] == false) {
                if (miners[i]->deviceResultsReady()) {
                    miners[i]->hostProcessResults();
                    minerIdle[i] = true;
                }
            }
            if (minerIdle[i]) {
                nIdle++;
            }
        }

        // if no miner is idle, wait a bit, to avoid polling too much GPUs about tasks status
        // same thing during a cpu block
        if (nIdle == 0 || cpuBlock) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // wait for updater thread to end (actually this will never happen ...)
    t.join();

    return true;
}
