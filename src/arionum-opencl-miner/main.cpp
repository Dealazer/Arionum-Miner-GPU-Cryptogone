#include <iostream>
#include <vector>
#include "../../argon2-gpu/include/commandline/commandlineparser.h"
#include "../../argon2-gpu/include/commandline/argumenthandlers.h"

#include "../../argon2-gpu/include/argon2-opencl/globalcontext.h"

#include "../../include/openclminer.h"
#include "../../include/cudaminer.h"

using namespace argon2;
using namespace std;
using namespace libcommandline;

struct OpenCLArguments {
    bool showHelp = false;
    bool listDevices = false;
    std::size_t deviceIndex = 0;
    int batchSize = 200;
    string address;
    string poolUrl;
};

void printDeviceList();

CommandLineParser<OpenCLArguments> buildCmdLineParser();


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

    std::cout << "Hello, World!" << std::endl;

    int i = 333;
    OpenClMiner(&i).mine();

    int j = 444;
    CudaMiner(&j).mine();
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
                    makeNumericHandler<OpenCLArguments, std::size_t>([](OpenCLArguments &state, std::size_t index) {
                        state.deviceIndex = (std::size_t) index;
                    }), "address", 'a', "public arionum address", "", "ADDRESS"),

            new ArgumentOption<OpenCLArguments>(
                    makeNumericHandler<OpenCLArguments, std::size_t>([](OpenCLArguments &state, std::size_t index) {
                        state.deviceIndex = (std::size_t) index;
                    }), "pool", 'p', "pool URL", "http://aropool.com", "POOL_URL"),

            new ArgumentOption<OpenCLArguments>(
                    makeNumericHandler<OpenCLArguments, std::size_t>([](OpenCLArguments &state, std::size_t index) {
                        state.deviceIndex = (std::size_t) index;
                    }), "device", 'd', "use device with index INDEX", "0", "INDEX"),

            new ArgumentOption<OpenCLArguments>(
                    makeNumericHandler<OpenCLArguments, int>([](OpenCLArguments &state, int index) {
                        state.batchSize = (int) index;
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