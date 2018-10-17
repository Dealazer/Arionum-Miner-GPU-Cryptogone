//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include <iostream>
#include <iomanip>

#include "../../include/openclminer.h"
#include "../../include/perfscope.h"

using namespace argon2;
using namespace std;
using argon2::t_optParams;

#define USE_PROGRAM_CACHE (1)
#if USE_PROGRAM_CACHE
#include <map>
static std::map<cl_device_id, argon2::opencl::ProgramContext*> s_programCache;
#endif

namespace cl {
    bool s_logCLErrors = true;
}

std::string toGB(size_t size) {
    double GB = (double)size / (1024.f * 1024.f * 1024.f);
    ostringstream os;
    os << std::fixed << std::setprecision(3) << GB << " GB";
    return os.str();
}

OpenClMiner::OpenClMiner(
    argon2::opencl::ProgramContext *progCtx, cl::CommandQueue & refQueue, 
    argon2::MemConfig memoryConfig,
    Stats *pStats, MinerSettings &settings, Updater *pUpdater) :
    Miner(memoryConfig, pStats, settings, pUpdater),
    progCtx(progCtx),
    queue(refQueue)
{
    const auto INITIAL_BLOCK_TYPE = BLOCK_GPU;

    t_optParams optPrms = configureArgon(
        Miner::getPasses(INITIAL_BLOCK_TYPE),
        Miner::getMemCost(INITIAL_BLOCK_TYPE),
        Miner::getLanes(INITIAL_BLOCK_TYPE));

    unit = new argon2::opencl::ProcessingUnit(
                queue,
                progCtx,
                params,
                memConfig,
                optPrms,
                INITIAL_BLOCK_TYPE);
}

void OpenClMiner::reconfigureArgon(
    uint32_t t_cost, uint32_t m_cost, uint32_t lanes) {
    if (!needReconfigure(t_cost, m_cost, lanes))
        return;

    t_optParams optPrms =
        configureArgon(t_cost, m_cost, lanes);

    unit->reconfigureArgon(params, optPrms, (t_cost == 1) ? BLOCK_CPU : BLOCK_GPU);
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
    PerfScope p("deviceResultsReady()");

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

std::string OpenClMiner::getInfo()
{
    std::ostringstream oss;
    auto &pCPUSizes = this->memConfig.batchSizes[BLOCK_CPU];
    auto &pGPUSizes = this->memConfig.batchSizes[BLOCK_GPU];

    oss << "CPU: ";
    auto mode = getMode(1, 1, 1);
    if (mode == PRECOMPUTE_LOCAL_STATE)
        oss << "(LOCAL_STATE) ";
    else if (mode == PRECOMPUTE_SHUFFLE)
        oss << "(SHUFFLE_BUF) ";
    else
        oss << "(BASELINE) ";
    oss << "(";
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        oss << pCPUSizes[i];
        oss << ((i == (MAX_BLOCKS_BUFFERS - 1)) ? ")" : " ");
    }    

    oss << ", ";

    oss << "GPU: (";
    for (int i = 0; i < MAX_BLOCKS_BUFFERS; i++) {
        oss << pGPUSizes[i];
        oss << ((i == (MAX_BLOCKS_BUFFERS - 1)) ? ")" : " ");
    }

    return oss.str();
}

OpenClMiningDevice::OpenClMiningDevice(
    size_t deviceIndex,
    uint32_t nTasks, uint32_t batchSizeGPU)
{
    // only support up to 4 buffers for now
    if (nTasks > MAX_BLOCKS_BUFFERS) {
        std::ostringstream oss;
        oss << "-t value must be <= " << nTasks;
        throw std::logic_error(oss.str());
    }

    // compute GPU blocks mem cost
    auto tGPU = Miner::getPasses(BLOCK_GPU);
    auto mGPU = Miner::getMemCost(BLOCK_GPU);
    auto lGPU = Miner::getLanes(BLOCK_GPU);
    argon2::Argon2Params paramsBlockGPU(
        32, "cifE2rK4nvmbVgQu", 16, nullptr, 0, nullptr, 0, tGPU, mGPU, lGPU);
    uint32_t blocksPerHashGPU = paramsBlockGPU.getMemoryBlocks();
    size_t memPerHashGPU = blocksPerHashGPU * ARGON2_BLOCK_SIZE;
    size_t memPerTaskGPU = batchSizeGPU * memPerHashGPU;

    // compute CPU blocks mem cost
    auto tCPU = Miner::getPasses(BLOCK_CPU);
    auto mCPU = Miner::getMemCost(BLOCK_CPU);
    auto lCPU = Miner::getLanes(BLOCK_CPU);
    argon2::Argon2Params paramsBlockCPU(
        32, "0KVwsNr6yT42uDX9", 16, nullptr, 0, nullptr, 0, tCPU, mCPU, lCPU);
    t_optParams optPrmsCPU = Miner::precomputeArgon(&paramsBlockCPU);
    optPrmsCPU.mode = PRECOMPUTE_SHUFFLE;
    uint32_t blocksPerHashCPU = optPrmsCPU.customBlockCount;
    size_t memPerHashCPU = blocksPerHashCPU * ARGON2_BLOCK_SIZE;

    // create context
    argon2::opencl::GlobalContext global;
    auto &devices = global.getAllDevices();
    auto device = &devices[deviceIndex];

#if USE_PROGRAM_CACHE
    cl_device_id deviceID = device->getCLDevice()();
    auto it = s_programCache.find(deviceID);
    if (it == s_programCache.end()) {
        s_programCache.insert(
            std::make_pair(
                deviceID,
                new argon2::opencl::ProgramContext(
                    &global, { *device }, ARGON_TYPE, ARGON_VERSION,
                    "./argon2-gpu/data/kernels/")));
    }
    progCtx = s_programCache[deviceID];
#else
    progCtx = new argon2::opencl::ProgramContext(
        &global, { *device }, type, version,
        "./argon2-gpu/data/kernels/");
#endif
    auto context = progCtx->getContext();

    // create queues
    for (uint32_t i = 0; i < nTasks; i++) {
        queues.emplace_back(context, device->getCLDevice(), CL_QUEUE_PROFILING_ENABLE);
    }

    // utility to create a buffer
    auto allocBuffer = [&](
        size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE) -> cl::Buffer {
        // allocate buffer
        cl::Buffer buf(context, flags, size);
        // warm it up
        uint8_t dummy = 0xFF;
        queues[0].enqueueWriteBuffer(buf, true, size-1, 1, &dummy);
        return buf;
    };

    // create nTasks block buffers
    for (uint32_t i = 0; i < nTasks; i++) {
        buffers.push_back(allocBuffer(memPerTaskGPU));
    }

    // create index buffer
    size_t indexSize = 0;
    if (optPrmsCPU.mode == argon2::PRECOMPUTE_LOCAL_STATE ||
        optPrmsCPU.mode == argon2::PRECOMPUTE_SHUFFLE) {
        indexSize = optPrmsCPU.customIndexNbSteps * 3 * sizeof(cl_uint);
        indexBuffer = allocBuffer(indexSize, CL_MEM_READ_ONLY);
        queues[0].enqueueWriteBuffer(
            indexBuffer, true, 0, indexSize, optPrmsCPU.customIndex);
    }

    // create mem configs for GPU tasks
    for (uint32_t i = 0; i < nTasks; i++) {
        MemConfig mc;
        mc.batchSizes[BLOCK_GPU][0] = batchSizeGPU;
        mc.blocksBuffers[BLOCK_GPU][0] = &buffers[i];

#define USE_SINGLE_TASK_FOR_CPU_BLOCKS (1)
#if USE_SINGLE_TASK_FOR_CPU_BLOCKS
        if (i == 0) {
            for (uint32_t j = 0; j < nTasks; j++) {
                mc.batchSizes[BLOCK_CPU][j] = memPerTaskGPU / memPerHashCPU;
                mc.blocksBuffers[BLOCK_CPU][j] = &buffers[j];
            }
            mc.indexBuffer = &indexBuffer;
        }
#else
        mc.batchSizes[BLOCK_CPU][0] = memPerTaskGPU / memPerHashCPU;
        mc.blocksBuffers[BLOCK_CPU][0] = &buffers[i];
        mc.indexBuffer = &indexBuffer;
#endif

        uint32_t totalNonces = (uint32_t)std::max(
            mc.getTotalHashes(BLOCK_GPU),
            mc.getTotalHashes(BLOCK_CPU));

        const size_t MAX_LANES = 4;
        const size_t IN_BLOCKS_MAX_SIZE = MAX_LANES * 2 * ARGON2_BLOCK_SIZE;
        mc.in = IN_BLOCKS_MAX_SIZE * totalNonces;

        const size_t OUT_BLOCKS_MAX_SIZE = MAX_LANES * ARGON2_BLOCK_SIZE;
        mc.out = OUT_BLOCKS_MAX_SIZE * totalNonces;

        minersConfigs.push_back(mc);
    }
}

argon2::MemConfig OpenClMiningDevice::getMemConfig(int taskId) {
    return minersConfigs[taskId];
}

#if 0

const size_t ARGON2_QWORDS_IN_BLOCK = ARGON2_BLOCK_SIZE / 8;

typedef struct block_ { uint64_t v[ARGON2_QWORDS_IN_BLOCK]; } block;

void copy_block(block *dst, const block *src) {
    memcpy(dst->v, src->v, sizeof(uint64_t) * ARGON2_QWORDS_IN_BLOCK);
}

void xor_block(block *dst, const block *src) {
    int i;
    for (i = 0; i < ARGON2_QWORDS_IN_BLOCK; ++i) {
        dst->v[i] ^= src->v[i];
    }
}

uint64_t fBlaMka(uint64_t x, uint64_t y);

#define G(a, b, c, d)                                                          \
    do {                                                                       \
        a = fBlaMka(a, b);                                                     \
        d = rotr64(d ^ a, 32);                                                 \
        c = fBlaMka(c, d);                                                     \
        b = rotr64(b ^ c, 24);                                                 \
        a = fBlaMka(a, b);                                                     \
        d = rotr64(d ^ a, 16);                                                 \
        c = fBlaMka(c, d);                                                     \
        b = rotr64(b ^ c, 63);                                                 \
    } while ((void)0, 0)

#define BLAKE2_ROUND_NOMSG(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,   \
                           v12, v13, v14, v15)                                 \
    do {                                                                       \
        G(v0, v4, v8, v12);                                                    \
        G(v1, v5, v9, v13);                                                    \
        G(v2, v6, v10, v14);                                                   \
        G(v3, v7, v11, v15);                                                   \
        G(v0, v5, v10, v15);                                                   \
        G(v1, v6, v11, v12);                                                   \
        G(v2, v7, v8, v13);                                                    \
        G(v3, v4, v9, v14);                                                    \
    } while ((void)0, 0)

static uint64_t rotr64(const uint64_t w, const unsigned c) {
    return (w >> c) | (w << (64 - c));
}

uint64_t fBlaMka(uint64_t x, uint64_t y) {
    const uint64_t m = UINT64_C(0xFFFFFFFF);
    const uint64_t xy = (x & m) * (y & m);
    return x + y + 2 * xy;
}

static void fG(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d) {
    a = fBlaMka(a, b);
    d = rotr64(d ^ a, 32);
    c = fBlaMka(c, d);
    b = rotr64(b ^ c, 24);
    a = fBlaMka(a, b);
    d = rotr64(d ^ a, 16);
    c = fBlaMka(c, d);
    b = rotr64(b ^ c, 63);
}

const int N_THREADS = 32;

std::atomic<size_t> s_thCpt = { 0 };

void waitCpt(int step) {
    s_thCpt++;
    while (s_thCpt.load() < step * N_THREADS) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(1));
    }
}

const uint8_t OFFS[4][16] = {
    {
        0, 4, 8, 12,
        0, 5, 10, 15,
        0, 32, 64, 96,
        0, 33, 80, 113
    },
    {
        1, 5, 9, 13,
        1, 6, 11, 12,
        1, 33, 65, 97,
        1, 48, 81, 96
    },
    {
        2, 6, 10, 14,
        2, 7, 8, 13,
        16, 48, 80, 112,
        16, 49, 64, 97,
    },
    {
        3, 7, 11, 15,
        3, 4, 9, 14,
        17, 49, 81, 113,
        17, 32, 65, 112,
    }
};

void fill_block_th(block &blockR, int threadID) {
    int thGroup = threadID / 4; // [0-7]
    int subId = threadID % 4;

#if 1
    auto offs = OFFS[subId];
    auto* p1 = blockR.v + 16 * thGroup;
    auto* p2 = blockR.v + 2 * thGroup;

    int step = 1;
    waitCpt(step++);

    fG(p1[offs[0]], p1[offs[1]], p1[offs[2]], p1[offs[3]]);
    waitCpt(step++);
    
    fG(p1[offs[4]], p1[offs[5]], p1[offs[6]], p1[offs[7]]);
    waitCpt(step++);

    fG(p2[offs[8]], p2[offs[9]], p2[offs[10]], p2[offs[11]]);
    waitCpt(step++);

    fG(p2[offs[12]], p2[offs[13]], p2[offs[14]], p2[offs[15]]);
    waitCpt(step++);
#else
    waitCpt(1);

    auto* p1 = blockR.v + 16 * thGroup;
    if (subId == 0)
        fG(p1[0], p1[4], p1[8], p1[12]);
    else if (subId == 1)
        fG(p1[1], p1[5], p1[9], p1[13]);
    else if (subId == 2)
        fG(p1[2], p1[6], p1[10], p1[14]);
    else if (subId == 3)
        fG(p1[3], p1[7], p1[11], p1[15]);

    waitCpt(2);

    if (subId == 0)
        fG(p1[0], p1[5], p1[10], p1[15]);
    else if (subId == 1)
        fG(p1[1], p1[6], p1[11], p1[12]);
    else if (subId == 2)
        fG(p1[2], p1[7], p1[8], p1[13]);
    else if (subId == 3)
        fG(p1[3], p1[4], p1[9], p1[14]);

    waitCpt(3);

    auto* p2 = blockR.v + 2 * thGroup;
    if (subId == 0)
        fG(p2[0], p2[32], p2[64], p2[96]);
    else if (subId == 1)
        fG(p2[1], p2[33], p2[65], p2[97]);
    else if (subId == 2)
        fG(p2[16], p2[48], p2[80], p2[112]);
    else if (subId == 3)
        fG(p2[17], p2[49], p2[81], p2[113]);

    waitCpt(4);

    if (subId == 0)
        fG(p2[0], p2[33], p2[80], p2[113]);
    else if (subId == 1)
        fG(p2[1], p2[48], p2[81], p2[96]);
    else if (subId == 2)
        fG(p2[16], p2[49], p2[64], p2[97]);
    else if (subId == 3)
        fG(p2[17], p2[32], p2[65], p2[112]);

    waitCpt(5);
#endif
}

static void fill_block(const block *prev_block, const block *ref_block,
    block *next_block, int with_xor) {
    block blockR, block_tmp;

    copy_block(&blockR, ref_block);
    xor_block(&blockR, prev_block);
    copy_block(&block_tmp, &blockR);
    /* Now blockR = ref_block + prev_block and block_tmp = ref_block + prev_block */
    if (with_xor) {
        /* Saving the next block contents for XOR over: */
        xor_block(&block_tmp, next_block);
        /* Now blockR = ref_block + prev_block and
        block_tmp = ref_block + prev_block + next_block */
    }

#if 1
    s_thCpt = 0;
    std::vector<std::thread> threads;
    for (int i = 0; i < N_THREADS; i++) {
        threads.emplace_back(
            fill_block_th, std::ref(blockR), i);
    }
    for (auto& it : threads) {
        it.join();
    }
#elif 1
    // block is 1024 bytes, or 128 uint64_t
    // we treat it as a 8 * 8 matrix of 16 bytes elements (2 uint64_t)

    // opencl has 32 threads
    // each thread processes 32 bytes (4 uint64_t)

    for (int i = 0; i < 8; ++i) {
        // at each step we process 8x 16 bytes elmts
        // so that is 16x uint64_t
        auto* p = blockR.v + 16 * i;
        fG(p[0], p[4], p[8],  p[12]);
        fG(p[1], p[5], p[9],  p[13]);
        fG(p[2], p[6], p[10], p[14]);
        fG(p[3], p[7], p[11], p[15]);

        fG(p[0], p[5], p[10], p[15]);
        fG(p[1], p[6], p[11], p[12]);
        fG(p[2], p[7], p[8],  p[13]);
        fG(p[3], p[4], p[9],  p[14]);
    }

    for (int i = 0; i < 8; ++i) {
        auto* p = blockR.v + 2 * i;
        fG(p[0], p[32], p[64], p[96]);
        fG(p[1], p[33], p[65], p[97]);
        fG(p[16], p[48], p[80], p[112]);
        fG(p[17], p[49], p[81], p[113]);

        fG(p[0], p[33], p[80], p[113]);
        fG(p[1], p[48], p[81], p[96]);
        fG(p[16], p[49], p[64], p[97]);
        fG(p[17], p[32], p[65], p[112]);
    }
#else
    /* Apply Blake2 on columns of 64-bit words: (0,1,...,15) , then
    (16,17,..31)... finally (112,113,...127) */
    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND_NOMSG(
            blockR.v[16 * i], blockR.v[16 * i + 1], blockR.v[16 * i + 2], blockR.v[16 * i + 3],
            blockR.v[16 * i + 4], blockR.v[16 * i + 5], blockR.v[16 * i + 6], blockR.v[16 * i + 7],
            blockR.v[16 * i + 8], blockR.v[16 * i + 9], blockR.v[16 * i + 10], blockR.v[16 * i + 11],
            blockR.v[16 * i + 12], blockR.v[16 * i + 13], blockR.v[16 * i + 14], blockR.v[16 * i + 15]);
    }

    /* Apply Blake2 on rows of 64-bit words: (0,1,16,17,...112,113), then
    (2,3,18,19,...,114,115).. finally (14,15,30,31,...,126,127) */
    for (i = 0; i < 8; i++) {
        BLAKE2_ROUND_NOMSG(
            blockR.v[2 * i],      blockR.v[2 * i + 1],  blockR.v[2 * i + 16],  blockR.v[2 * i + 17],
            blockR.v[2 * i + 32], blockR.v[2 * i + 33], blockR.v[2 * i + 48],  blockR.v[2 * i + 49],
            blockR.v[2 * i + 64], blockR.v[2 * i + 65], blockR.v[2 * i + 80],  blockR.v[2 * i + 81],
            blockR.v[2 * i + 96], blockR.v[2 * i + 97], blockR.v[2 * i + 112], blockR.v[2 * i + 113]);
    }
#endif

    copy_block(next_block, &block_tmp);
    xor_block(next_block, &blockR);
}

#pragma warning(disable:4996)

void dumpBlock(block &b, char* filename, char* dataName) {
    FILE* f = fopen(filename, "w");
    fprintf(f, "block %s = { { \n", dataName);
    for (uint64_t i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        fprintf(f, "UINT64_C(%llu),\n", b.v[i]);
    }
    fprintf(f, "} };\n");
    fclose(f);
}

#include "../../testRef.h"

void checkBlock(block &b, block &ref) {
    for (uint64_t i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        if (b.v[i] != ref.v[i]) {
            printf("block check failed, i=%llu\n", i);
            exit(1);
        }
    }
}

void OpenClMiningDevice::testArgon() {
    block prev, ref, next;
    for (uint64_t i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++) {
        prev.v[i] = i * 19345678;
        ref.v[i] = i * 8975433;
        next.v[i] = 0;
    }

    //dumpBlock(next, "prev.h", "s_blockPrev");
    //dumpBlock(next, "ref.h", "s_blockRef");

    fill_block(&prev, &ref, &next, false);
    //dumpBlock(next, "testRef.h", "s_blockTest");
    dumpBlock(next, "result.h", "s_result");

    checkBlock(next, s_blockTest);
}

#endif