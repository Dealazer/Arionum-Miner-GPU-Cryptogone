#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include "../../argon2-gpu/include/commandline/commandlineparser.h"
#include "../../argon2-gpu/include/commandline/argumenthandlers.h"


#include <iomanip>
#include <boost/algorithm/string.hpp>

#ifdef _MSC_VER
#include "../../include/win_tools.h"
#endif

#include "../../include/miner_version.h"

#include <windows.h>

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
    size_t deviceIndex = 0;
    std::vector<int> deviceIndexList;
    size_t batchSize = 1;
    string address = "4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL";
    string poolUrl = "http://aropool.com";
    size_t threadsPerDevice = 1;
    double d = 1;
};

template <class CONTEXT, class MINER>
void parseDeviceArgs(const OpenCLArguments &args, vector<Miner *> &miners, Stats* stats, Updater *updater, MinerSettings& settings) {
    // -u option
    if (args.allDevices) {
        cout << "Use all Devices" << endl;
        CONTEXT global;
        auto &devices = global.getAllDevices();
        for (size_t i = 0; i < devices.size(); ++i) {
            for (int j = 0; j < args.threadsPerDevice; ++j) {
                Miner *miner = new MINER(stats, &settings, updater, &i);
                miners.push_back(miner);
            }
        }

    }
    // -k option
    else if (args.deviceIndexList.size() > 0) {
        auto global = new CONTEXT();
        auto &devices = global->getAllDevices();
        for (const auto &it : args.deviceIndexList) {
            if (it >= devices.size())
                continue;
            size_t deviceIndex = it;
            cout << "--- start using device #" << deviceIndex << " (-k)" << endl;
            for (int j = 0; j < args.threadsPerDevice; ++j) {
                Miner *miner = new MINER(stats, &settings, updater, &deviceIndex);
                miners.push_back(miner);
            }
        }
    } // -d option
    else {
        size_t deviceIndex = args.deviceIndex;
        cout << "--- start using device #" << deviceIndex << " (-d)" << endl;
        for (int j = 0; j < args.threadsPerDevice; ++j) {
            Miner *miner = new MINER(stats, &settings, updater, &deviceIndex);
            miners.push_back(miner);
        }
    }
}

CommandLineParser<OpenCLArguments> buildCmdLineParser() {
    static const auto positional = PositionalArgumentHandler<OpenCLArguments>(
        [](OpenCLArguments &, const std::string &) {});

    std::vector<const CommandLineOption<OpenCLArguments> *> options{
        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.listDevices = true; },
            "list-devices", 'l', "list all available devices and exit"),

        new ArgumentOption<OpenCLArguments>(
            makeNumericHandler<OpenCLArguments, std::size_t>([](OpenCLArguments &state, std::size_t index) {
        state.deviceIndex = (std::size_t) index;
    }), "device", 'd', "use device with index INDEX", "0", "INDEX"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string deviceList) {
        std::vector<string> indicesStr;
        boost::split(indicesStr, deviceList, boost::is_any_of(","));
        for (auto& it : indicesStr) {
            auto idStr = boost::trim_left_copy(it);
            int id;
            auto count = sscanf_s(idStr.c_str(), "%d", &id);
            if (count == 1) {
                state.deviceIndexList.push_back(id);
            }
            else {
                cout << "Error parsing device-list parameter (-k), ignoring it..." << endl;
                state.deviceIndexList.clear();
                break;
            }
        }
    }, "device-list", 'k', "use list of devices (comma separated, ex: -k 0,2,3)", "DEVICE LIST"),

        new FlagOption<OpenCLArguments>(
            [](OpenCLArguments &state) { state.allDevices = true; },
            "use-all-devices", 'u', "use all available devices"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string address) { state.address = address; }, "address", 'a',
            "public arionum address",
            "4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL",
            "ADDRESS"),

        new ArgumentOption<OpenCLArguments>(
            [](OpenCLArguments &state, const string poolUrl) { state.poolUrl = poolUrl; }, "pool", 'p',
            "pool URL", "http://aropool.com", "POOL_URL"),

        new ArgumentOption<OpenCLArguments>(
            makeNumericHandler<OpenCLArguments, double>([](OpenCLArguments &state, double devFee) {
        state.d = devFee <= 0.5 ? 1 : devFee;
    }), "dev-donation", 'D', "developer donation", "0.5", "PERCENTAGE"),

        new ArgumentOption<OpenCLArguments>(
            makeNumericHandler<OpenCLArguments, std::size_t>(
                [](OpenCLArguments &state, std::size_t threadsPerDevice) {
        state.threadsPerDevice = (std::size_t) threadsPerDevice;
    }), "threads-per-device", 't', "thread to use per device", "1", "THREADS"),

        new ArgumentOption<OpenCLArguments>(
            makeNumericHandler<OpenCLArguments, size_t>([](OpenCLArguments &state, size_t index) {
        state.batchSize = index;
    }), "batchSize", 'b', "batch size", "1", "SIZE"),

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
    for (size_t i = 0; i < devices.size(); i++) {
        auto &device = devices[i];
        cout << "Device #" << i << ": " << device.getName()
            << endl;
    }
}

int _gettimeofday(struct timeval* p, void* tz) {
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

string generateUniqid() {
    struct timeval tv {};
    _gettimeofday(&tv, nullptr);
    auto sec = (int)tv.tv_sec;
    auto usec = (int)(tv.tv_usec % 0x100000);
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(8) << std::hex << sec << std::setfill('0') << std::setw(5) << std::hex
        << usec;
    return ss.str();
}

template <class CONTEXT, class MINER>
int commonMain(const char *const *argv) {
#ifdef _MSC_VER
    // set a fixed console size (default is not wide enough)
    setConsoleSize(150, 40, 2000);
#endif

    cout << getVersionStr() << endl << endl;

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
    string uniqid = generateUniqid();
    MinerSettings settings(&args.poolUrl, &args.address, &uniqid, &args.batchSize);
    auto *stats = new Stats(args.d);

    // show launch settings in console
    std::cout << settings << std::endl;

    // launch updater thread
    auto *updater = new Updater(stats, &settings);
    updater->update();
    thread t(&Updater::start, updater);

    // chooses GPU devices to run on & create miners
    vector<Miner *> miners;
    parseDeviceArgs<CONTEXT, MINER>(args, miners, stats, updater, settings);

    // launch miner threads
    vector<thread> threads;
    for (auto const &miner : miners) {
        thread minerT(&Miner::mine, miner);
        threads.push_back(std::move(minerT));
    }

    // wait for miner threads to end
    for (auto &thread : threads) {
        thread.join();
    }

    // wait for updater thread to end
    t.join();

    return true;
}
