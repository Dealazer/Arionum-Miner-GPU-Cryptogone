//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include <iostream>
#include <iomanip>

#include "../../include/openclminer.h"

using namespace argon2;
using namespace std;
using argon2::t_optParams;
using argon2::PRECOMPUTE;
using argon2::BASELINE;

#define USE_PROGRAM_CACHE (1)

#if USE_PROGRAM_CACHE
#include <map>
static std::map<cl_device_id, argon2::opencl::ProgramContext*> s_programCache;
#endif

OpenClMiner::OpenClMiner(
    size_t deviceIndex, uint32_t batchSizeGPU,
    Stats *pStats, MinerSettings &settings, Updater *pUpdater) :
    Miner(batchSizeGPU, pStats, settings, pUpdater) {

    global = new argon2::opencl::GlobalContext();

    auto &devices = global->getAllDevices();
    device = &devices[deviceIndex];

#if USE_PROGRAM_CACHE
    cl_device_id deviceID = device->getCLDevice()();
    auto it = s_programCache.find(deviceID);
    if (it == s_programCache.end()) {
        s_programCache.insert(
            std::make_pair(
                deviceID,
                new argon2::opencl::ProgramContext(
                    global, { *device }, type, version,
                    "./argon2-gpu/data/kernels/")));
    }
    progCtx = s_programCache[deviceID];
#else
    progCtx = new argon2::opencl::ProgramContext(
        global, {*device}, type, version,
        "./argon2-gpu/data/kernels/");
#endif

//    auto context = progCtx->getContext();
//    configure_queue = cl::CommandQueue(context, device->getCLDevice(), 0);
}

#define VERBOSE_CONFIGURE (1)

std::string toGB(size_t size) {
    double GB = (double)size / (1024.f * 1024.f * 1024.f);
    ostringstream os;
    os << std::fixed << std::setprecision(3) << GB << " GB";
    return os.str();
}

void printBlockBuffer(
    const argon2::MemConfig& mc,
    int i) {
    auto size = mc.blocksBuffers[i];
    if (size == 0) {
        return;
    }
    cout
        << "buffer " << i << endl
        << "  size       = "
        << toGB(size) << endl
        << "  nHashesCPU = "
        << mc.batchSizes[BLOCK_CPU][i] << endl
        << "  nHashesGPU = "
        << mc.batchSizes[BLOCK_GPU][i] << endl;
}

namespace cl {
    bool s_logCLErrors = true;
}

bool OpenClMiner::testAlloc(size_t size)
{
    bool ok = true;
    {
        bool prev = cl::s_logCLErrors;
        cl::s_logCLErrors = false;
        try {
            auto context = progCtx->getContext();
            cl::Buffer testBuf(context, CL_MEM_READ_WRITE, size);
            if (testBuf() == 0)
                ok = false;
            uint8_t dummy;
            configure_queue.enqueueWriteBuffer(testBuf, true, size - 1, 1, &dummy);
        }
        catch (exception e) {
            ok = false;
        }
        cl::s_logCLErrors = prev;
    }
    return ok;
}

size_t OpenClMiner::findMaxAlloc(size_t maxMem) {
    auto cl_dev = device->getCLDevice();
    size_t min = 0;
    size_t max = !maxMem ?
        cl_dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() : maxMem;
    size_t best = 0;

    while (1) {
        size_t cur = (max + min) / 2;
        if (!testAlloc(cur))
            max = cur - 1;
        else {
            best = cur;
            min = cur + 1;
        }
        if (min >= max) {
            return testAlloc(max) ? max : best;
        }
    }
    return 0;
}

argon2::MemConfig OpenClMiner::configure(uint32_t batchSizeGPU) {
    argon2::MemConfig mc;

    // get device mem info
    auto cl_dev = device->getCLDevice();
    size_t deviceTotalMem = cl_dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    size_t maxAllocCL = cl_dev.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    auto deviceNComputeUnits = cl_dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    auto deviceMaxWorkGroupSize = cl_dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    // cannot use more mem than device has
    //maxMemUsage = std::min(maxMemUsage, deviceTotalMem);

    // get mem per hash needed for CPU round
    auto optParamsCPU = configureArgon(
        Miner::getPasses(BLOCK_CPU),
        Miner::getMemCost(BLOCK_CPU),
        Miner::getLanes(BLOCK_CPU));
    uint32_t cpuBlocksPerHash = (optParamsCPU.mode == PRECOMPUTE) ?
        optParamsCPU.customBlockCount :
        params->getMemoryBlocks();
    size_t memPerHashCPU = cpuBlocksPerHash * ARGON2_BLOCK_SIZE;

    // get mem per hash needed for GPU round
    auto optParamsGPU = configureArgon(
        Miner::getPasses(BLOCK_GPU),
        Miner::getMemCost(BLOCK_GPU),
        Miner::getLanes(BLOCK_GPU));
    uint32_t gpuBlocksPerHash = params->getMemoryBlocks();
    size_t memPerHashGPU = gpuBlocksPerHash * ARGON2_BLOCK_SIZE;

    // get mem needed for precomputed indices
    mc.index = 0;
    if (optParamsCPU.mode == PRECOMPUTE) {
        mc.index = optParamsCPU.customIndexNbSteps * 3 * sizeof(cl_uint);
    }

    //
    size_t maxTotalNonces = batchSizeGPU;

    // get max mem needed for sending nonces
    const size_t MAX_LANES = 4;
    const size_t IN_BLOCKS_MAX_SIZE = MAX_LANES * 2 * ARGON2_BLOCK_SIZE;
    mc.in = IN_BLOCKS_MAX_SIZE * maxTotalNonces;
    //mc.in = 1835008;

    // get max mem needed for getting back results
    const size_t OUT_BLOCKS_MAX_SIZE = MAX_LANES * ARGON2_BLOCK_SIZE;
    mc.out = OUT_BLOCKS_MAX_SIZE * maxTotalNonces;
    //mc.out = 917504;

    // now we can start configuring buffers
    auto setBlockBuffer = [&](int i, size_t size) -> void {
        mc.blocksBuffers[i] = (uint32_t)size;
        mc.batchSizes[BLOCK_CPU][i] = size / memPerHashCPU;
        mc.batchSizes[BLOCK_GPU][i] = size / memPerHashGPU;
    };

    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++)
        setBlockBuffer(i, 0);

    // force test
    size_t memBlocks = 0;
    size_t bufSize;

    bufSize = batchSizeGPU * memPerHashGPU;
    setBlockBuffer(0, bufSize);
    memBlocks += bufSize;

#if VERBOSE_CONFIGURE
    auto memMisc = mc.index + mc.in + mc.out;
    auto memBlocksMax = deviceTotalMem - memMisc;
    auto maxCPUHashes = memBlocksMax / memPerHashCPU;
    auto maxGPUHashes = memBlocksMax / memPerHashGPU;
    cout
        << "deviceTotalMem   : " << toGB(deviceTotalMem) << endl
        << "computeUnits     : " << deviceNComputeUnits << endl
        << "maxWorkGroupSize : " << deviceMaxWorkGroupSize << endl
        << "maxAllocCL       : " << toGB(maxAllocCL) << endl
        << "memBlocks        : " << toGB(memBlocks) << endl
        << "memMisc          : " << toGB(memMisc) << endl
        << "theorical        : "
        << "CPU " << maxCPUHashes
        << ", GPU " << maxGPUHashes << endl
        << "current          : "
        << "CPU " << mc.getTotalHashes(BLOCK_CPU)
        << ", GPU " << mc.getTotalHashes(BLOCK_GPU) << endl;
    for (int i = 0; i<MAX_BLOCKS_BUFFERS; i++) {
        printBlockBuffer(mc, i);
    }
#endif
    return mc;
}

bool OpenClMiner::createUnit() {
    const auto INITIAL_BLOCK_TYPE = BLOCK_GPU;
    
    t_optParams optPrms = configureArgon(
        Miner::getPasses(INITIAL_BLOCK_TYPE),
        Miner::getMemCost(INITIAL_BLOCK_TYPE),
        Miner::getLanes(INITIAL_BLOCK_TYPE));

    unit = new argon2::opencl::ProcessingUnit(
        device,
        progCtx,
        params,
        memConfig,
        optPrms,
        INITIAL_BLOCK_TYPE);

    return true;
}

void OpenClMiner::reconfigureArgon(
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes) {

    if (!needReconfigure(t_cost, m_cost, lanes))
        return;

    t_optParams optPrms =
        configureArgon(t_cost, m_cost, lanes);

    unit->reconfigureArgon(
        device,
        params,
        optPrms,
        (t_cost == 1) ? BLOCK_CPU : BLOCK_GPU);
}

void OpenClMiner::deviceUploadTaskDataAsync() {
#if (!OPEN_CL_SKIP_MEM_TRANSFERS)
    unit->uploadInputDataAsync(bases);
#endif
}

void OpenClMiner::deviceLaunchTaskAsync() {
    unit->runKernelAsync();
}

void OpenClMiner::deviceFetchTaskResultAsync() {
    unit->fetchResultsAsync();
}

void OpenClMiner::deviceWaitForResults() {
    unit->waitForResults();
}

bool OpenClMiner::deviceResultsReady() {
    bool queueFinished = unit->resultsReady();
    if (queueFinished) {
#if (!OPEN_CL_SKIP_MEM_TRANSFERS)
        auto blockType = (params->getLanes() == 1) ? 
            BLOCK_CPU : BLOCK_GPU;
        int curHash = 0;
        for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
            auto nHashes = memConfig.batchSizes[blockType][i];
            for (auto j = 0; j < nHashes; j++) {
                resultsPtrs[i][j] = unit->getResultPtr(curHash);
                curHash++;
            }
        }
#endif
    }
    return queueFinished;
}

