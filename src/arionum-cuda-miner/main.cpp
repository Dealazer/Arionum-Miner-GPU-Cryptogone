//
// Created by guli on 01/02/18.
//

#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include "../../argon2-gpu/include/commandline/commandlineparser.h"
#include "../../argon2-gpu/include/commandline/argumenthandlers.h"

#include "../../include/minerdata.h"
#include "../../include/updater.h"
#include "../../include/cudaminer.h"
//#include <sys/time.h>
#include <iomanip>
#include "../../include/simplecudaminer.h"

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
    size_t batchSize = 1;
    string address = "4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL";
    string poolUrl = "http://aropool.com";
    size_t threadsPerDevice = 1;
    double d = 1;
};

void printDeviceList();

CommandLineParser<OpenCLArguments> buildCmdLineParser();

string generateUniqid();


int main(int, const char *const *argv) {
    CommandLineParser<OpenCLArguments> parser = buildCmdLineParser();
    OpenCLArguments args;
    int ret = parser.parseArguments(args, argv);
    if (ret != 0) {
        return ret;
    }
    if (args.showHelp) {
        parser.printHelp(argv);
        return 0;
    }
    if (args.listDevices) {
        printDeviceList();
        return 0;
    }

    string uniqid = generateUniqid();
    MinerSettings settings(&args.poolUrl, &args.address, &uniqid, &args.batchSize);

    std::cout << settings << std::endl;

    vector<Miner *> miners;
    auto *stats = new Stats(args.d);

    auto *updater = new Updater(stats, &settings);
    updater->update();

    thread t(&Updater::start, updater);

    if (args.allDevices) {
        cout << "Use all Devices" << endl;
        cuda::GlobalContext global;
        auto &devices = global.getAllDevices();
        for (size_t i = 0; i < devices.size(); ++i) {
            for (int j = 0; j < args.threadsPerDevice; ++j) {
                Miner *miner = new CudaMiner(stats, &settings, updater, &i);
                miners.push_back(miner);
            }
        }

    } else {
        size_t deviceIndex = args.deviceIndex;
        cout << "start using device #" << deviceIndex << endl;
        for (int j = 0; j < args.threadsPerDevice; ++j) {
            Miner *miner = new CudaMiner(stats, &settings, updater, &deviceIndex);
            miners.push_back(miner);
        }
    }
    vector<thread> threads;
    for (auto const &miner: miners) {
        thread minerT(&Miner::mine, miner);
        threads.push_back(std::move(minerT));
    }
    for (auto &thread : threads) {
        thread.join();
    }
    t.join();
    return 0;
}

CommandLineParser<OpenCLArguments> buildCmdLineParser() {
    static const auto positional = PositionalArgumentHandler<OpenCLArguments>(
            [](OpenCLArguments &, const std::string &) {});

    std::vector<const CommandLineOption<OpenCLArguments> *> options{
            new FlagOption<OpenCLArguments>(
                    [](OpenCLArguments &state) { state.listDevices = true; },
                    "list-devices", 'l', "list all available devices and exit"),

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
                    makeNumericHandler<OpenCLArguments, std::size_t>([](OpenCLArguments &state, std::size_t index) {
                        state.deviceIndex = (std::size_t) index;
                    }), "device", 'd', "use device with index INDEX", "0", "INDEX"),

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

void printDeviceList() {
    cuda::GlobalContext global;
    auto &devices = global.getAllDevices();
    for (size_t i = 0; i < devices.size(); i++) {
        auto &device = devices[i];
        cout << "Device #" << i << ": " << device.getName()
             << endl;
    }
}

#include <windows.h>

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
    struct timeval tv{};
    _gettimeofday(&tv, nullptr);
    auto sec = (int) tv.tv_sec;
    auto usec = (int) (tv.tv_usec % 0x100000);
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(8) << std::hex << sec << std::setfill('0') << std::setw(5) << std::hex
       << usec;
    return ss.str();
}