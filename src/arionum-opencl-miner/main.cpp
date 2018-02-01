#include <iostream>
#include <vector>
#include <thread>
#include <cstring>
#include "../../argon2-gpu/include/commandline/commandlineparser.h"
#include "../../argon2-gpu/include/commandline/argumenthandlers.h"

#include "../../argon2-gpu/include/argon2-opencl/globalcontext.h"

#include "../../include/openclminer.h"
#include "../../include/minerdata.h"
#include "../../include/updater.h"
#include "../../include/cudaminer.h"
#include <sys/time.h>
#include <iomanip>

using namespace argon2;
using namespace std;
using namespace libcommandline;

struct OpenCLArguments {
    bool showHelp = false;
    bool listDevices = false;
    size_t deviceIndex = 0;
    size_t batchSize = 1;
    string address = "4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL";
    string poolUrl = "http://aropool.com";
};

void printDeviceList();

CommandLineParser<OpenCLArguments> buildCmdLineParser();

string generateUniqid();


int main(int, const char *const *argv) {
    auto saltBytes = new char[16];

    argon2::opencl::GlobalContext global;
    auto &devices = global.getAllDevices();
    auto &device = devices[2];
    cout << "using device " << device.getName() << endl;
    argon2::Type type = argon2::ARGON2_I;
    argon2::Version version = argon2::ARGON2_VERSION_13;
    argon2::opencl::ProgramContext progCtx(&global, {device}, type, version);

    argon2::Argon2Params params(32, saltBytes, 16, nullptr, 0, nullptr, 0, 4, 16384, 4);
    argon2::opencl::ProcessingUnit unit(&progCtx, &params, &device, 1, false, false);


    if(true)
        return 0;



    MinerData data;
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
    auto *stats = new Stats();

    auto *updater = new Updater(stats, &settings, &data);
    updater->update();
    thread t(&Updater::start, updater);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    Miner *miner = new OpenClMiner(stats, &settings, &data, updater, &args.deviceIndex);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    miners.push_back(miner);
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

            new ArgumentOption<OpenCLArguments>(
                    [](OpenCLArguments &state, const string address) { state.address = address; }, "address", 'a',
                    "public arionum address",
                    "4hDFRqgFDTjy5okh2A7JwQ3MZM7fGyaqzSZPEKUdgwSM8sKLPEgs8Awpdgo3R54uo1kGMnxujQQpF94qV6SxEjRL",
                    "ADDRESS"),

            new ArgumentOption<OpenCLArguments>(
                    [](OpenCLArguments &state, const string poolUrl) { state.poolUrl = poolUrl; }, "pool", 'p',
                    "pool URL", "http://aropool.com", "POOL_URL"),

            new ArgumentOption<OpenCLArguments>(
                    makeNumericHandler<OpenCLArguments, std::size_t>([](OpenCLArguments &state, std::size_t index) {
                        state.deviceIndex = (std::size_t) index;
                    }), "device", 'd', "use device with index INDEX", "0", "INDEX"),

            new ArgumentOption<OpenCLArguments>(
                    makeNumericHandler<OpenCLArguments, size_t>([](OpenCLArguments &state, size_t index) {
                        state.batchSize = index;
                    }), "batchSize", 'b', "batch size", "200", "SIZE"),

            new FlagOption<OpenCLArguments>(
                    [](OpenCLArguments &state) { state.showHelp = true; },
                    "help", '?', "show this help and exit")
    };

    return CommandLineParser<OpenCLArguments>(
            "A tool for testing the argon2-opencl and argon2-cuda libraries.",
            positional, options);
}

void printDeviceList() {
    opencl::GlobalContext global;
    auto &devices = global.getAllDevices();
    for (size_t i = 0; i < devices.size(); i++) {
        auto &device = devices[i];
        cout << "Device #" << i << ": " << device.getName()
             << endl;
    }
}

string generateUniqid() {
    struct timeval tv{};
    gettimeofday(&tv, nullptr);
    auto sec = (int) tv.tv_sec;
    auto usec = (int) (tv.tv_usec % 0x100000);
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(8) << std::hex << sec << std::setfill('0') << std::setw(5) << std::hex
       << usec;
    return ss.str();
}