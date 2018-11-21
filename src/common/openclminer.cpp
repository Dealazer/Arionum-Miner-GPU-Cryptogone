//
// Created by guli on 31/01/18. Modified by Cryptogone (windows port, fork at block 80k, optimizations)
//

#include "../../include/openclminer.h"
#include "../../include/perfscope.h"

#include <iostream>
#include <iomanip>
#include <map>
#include <memory>

void OpenClMiningDevice::initialize(uint32_t deviceIndex) {
    device = globalCtx.getAllDevices()[deviceIndex];

    static std::map<
        cl_device_id,
        std::unique_ptr<argon2::opencl::ProgramContext>> s_programCache;

    cl_device_id deviceID = device.getCLDevice()();
    auto it = s_programCache.find(deviceID);
    if (it == s_programCache.end()) {
        s_programCache.emplace(
            deviceID,
            std::unique_ptr<argon2::opencl::ProgramContext>(
                new argon2::opencl::ProgramContext(
                    &globalCtx, { device }, ARGON_TYPE, ARGON_VERSION,
                    "./argon2-gpu/data/kernels/")));
    }

    progCtx = s_programCache[deviceID].get();
}

size_t OpenClMiningDevice::maxAllocSize() const {
    return 
        device.getCLDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
}

void OpenClMiningDevice::warmupBuffer(cl::Buffer &buf, size_t size) {
    uint8_t dummy = 0xFF;
    queue(0).enqueueWriteBuffer(buf, true, size - 1, 1, &dummy);
}

bool s_openCL_logErrors = true;

bool OpenClMiningDevice::testAlloc(size_t size) {
    bool ok;
    s_openCL_logErrors = false;
    try {
        const cl::Context& context = progCtx->getContext();
        std::unique_ptr<cl::Buffer> buf = 
            std::make_unique<cl::Buffer>(context, CL_MEM_READ_WRITE, size);
        warmupBuffer(*buf, size);
        ok = true;
    }
    catch (const std::exception &) {
        ok = false;
    }
    s_openCL_logErrors = true;
    return ok;
}

cl::Buffer* OpenClMiningDevice::newBuffer(size_t size) {
    // add a new buffer
    const cl::Context& context = progCtx->getContext();
    buffers.emplace_back(new cl::Buffer(context, CL_MEM_READ_WRITE, size));
    warmupBuffer(*buffers.back(), size);
    return buffers.back().get();
}

void OpenClMiningDevice::writeBuffer(cl::Buffer* buf, const void* src, size_t size) {
    queue(0).enqueueWriteBuffer(*buf, true, 0, size, src);
}

cl::CommandQueue* OpenClMiningDevice::newQueue() {
    queues.emplace_back(new cl::CommandQueue(
        progCtx->getContext(),
        device.getCLDevice(),
        CL_QUEUE_PROFILING_ENABLE));
    return queues.back().get();
}

OpenClMiner::OpenClMiner(
    argon2::opencl::ProgramContext *progCtx, cl::CommandQueue & refQueue,
    const argon2::MemConfig &memConfig, const Services& services,
    argon2::OPT_MODE gpuOptimizationMode) :
    AroMiner(memConfig, services, gpuOptimizationMode),
    ctx{refQueue, *progCtx},
    pUnit(new argon2::opencl::ProcessingUnit(miningConfig, ctx)) {
}

void OpenClMiner::reconfigureKernel() {
    pUnit->configure();
}

void OpenClMiner::uploadInputs_Async() {
    pUnit->uploadInputDataAsync(nonces.bases);
}

void OpenClMiner::run_Async() {
    pUnit->runKernelAsync();
}

void OpenClMiner::fetchResults_Async() {
    pUnit->fetchResultsAsync();
}

bool OpenClMiner::resultsReady() {
    PerfScope p("deviceResultsReady()");
    bool queueFinished = pUnit->resultsReady();
    return queueFinished;
}

uint8_t * OpenClMiner::resultsPtr() {
    return pUnit->results();
}

#if 0
std::string toGB(size_t size) {
    double GB = (double)size / (1024.f * 1024.f * 1024.f);
    ostringstream os;
    os << std::fixed << std::setprecision(3) << GB << " GB";
    return os.str();
}
#endif

//////////////////////////////////////////////////////////////////////////////////////

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
