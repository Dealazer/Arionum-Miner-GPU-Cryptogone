/*
 * Argon2 reference source code package - reference C implementations
 *
 * Copyright 2015
 * Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, and Samuel Neves
 *
 * You may use this work under the terms of a Creative Commons CC0 1.0
 * License/Waiver or the Apache Public License 2.0, at your option. The terms of
 * these licenses can be found at:
 *
 * - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
 * - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0
 *
 * You should have received a copy of both of these licenses along with this
 * software. If not, they may be obtained at the above URLs.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "argon2.h"
#include "core.h"

#include "blake2/blake2.h"
#include "blake2/blamka-round-opt.h"

/*
 * Function fills a new memory block and optionally XORs the old block over the new one.
 * Memory must be initialized.
 * @param state Pointer to the just produced block. Content will be updated(!)
 * @param ref_block Pointer to the reference block
 * @param next_block Pointer to the block to be XORed over. May coincide with @ref_block
 * @param with_xor Whether to XOR into the new block (1) or just overwrite (0)
 * @pre all block pointers must be valid
 */
#if defined(__AVX512F__)
static void fill_block(__m512i *state, const block *ref_block,
                       block *next_block, int with_xor) {
    __m512i block_XY[ARGON2_512BIT_WORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
            state[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i *)ref_block->v + i));
            block_XY[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i *)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm512_xor_si512(
                state[i], _mm512_loadu_si512((const __m512i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 2; ++i) {
        BLAKE2_ROUND_1(
            state[8 * i + 0], state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
            state[8 * i + 4], state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]);
    }

    for (i = 0; i < 2; ++i) {
        BLAKE2_ROUND_2(
            state[2 * 0 + i], state[2 * 1 + i], state[2 * 2 + i], state[2 * 3 + i],
            state[2 * 4 + i], state[2 * 5 + i], state[2 * 6 + i], state[2 * 7 + i]);
    }

    for (i = 0; i < ARGON2_512BIT_WORDS_IN_BLOCK; i++) {
        state[i] = _mm512_xor_si512(state[i], block_XY[i]);
        _mm512_storeu_si512((__m512i *)next_block->v + i, state[i]);
    }
}
#elif defined(__AVX2__)
static void fill_block_precompute(__m256i *state, const block *ref_block, block *next_block) {
    __m256i block_XY[ARGON2_HWORDS_IN_BLOCK];
    unsigned int i;

    for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
        block_XY[i] = state[i] = _mm256_xor_si256(
            state[i], _mm256_loadu_si256((const __m256i *)ref_block->v + i));
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND_1(state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
            state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]);
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND_2(state[0 + i], state[4 + i], state[8 + i], state[12 + i],
            state[16 + i], state[20 + i], state[24 + i], state[28 + i]);
    }

    if (next_block) {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            state[i] = _mm256_xor_si256(state[i], block_XY[i]);
            _mm256_storeu_si256((__m256i *)next_block->v + i, state[i]);
        }
    }
    else {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            state[i] = _mm256_xor_si256(state[i], block_XY[i]);
        }
    }
}

static void fill_block(__m256i *state, const block *ref_block,
                       block *next_block, int with_xor) {
    __m256i block_XY[ARGON2_HWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            state[i] = _mm256_xor_si256(
                state[i], _mm256_loadu_si256((const __m256i *)ref_block->v + i));
            block_XY[i] = _mm256_xor_si256(
                state[i], _mm256_loadu_si256((const __m256i *)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm256_xor_si256(
                state[i], _mm256_loadu_si256((const __m256i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND_1(state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
                       state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]);
    }

    for (i = 0; i < 4; ++i) {
        BLAKE2_ROUND_2(state[ 0 + i], state[ 4 + i], state[ 8 + i], state[12 + i],
                       state[16 + i], state[20 + i], state[24 + i], state[28 + i]);
    }

    for (i = 0; i < ARGON2_HWORDS_IN_BLOCK; i++) {
        state[i] = _mm256_xor_si256(state[i], block_XY[i]);
        _mm256_storeu_si256((__m256i *)next_block->v + i, state[i]);
    }
}
#else
static void fill_block_precompute(
    __m128i *state, const block *ref_block, block *next_block) 
{
    __m128i block_XY[ARGON2_OWORDS_IN_BLOCK];
    unsigned int i;

    for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
        block_XY[i] = state[i] = _mm_xor_si128(
            state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * i + 0], state[8 * i + 1], state[8 * i + 2],
            state[8 * i + 3], state[8 * i + 4], state[8 * i + 5],
            state[8 * i + 6], state[8 * i + 7]);
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i],
            state[8 * 3 + i], state[8 * 4 + i], state[8 * 5 + i],
            state[8 * 6 + i], state[8 * 7 + i]);
    }

    if (next_block) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = _mm_xor_si128(state[i], block_XY[i]);
            _mm_storeu_si128((__m128i *)next_block->v + i, state[i]);
        }
    }
    else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = _mm_xor_si128(state[i], block_XY[i]);
        }
    }
}

static void fill_block(__m128i *state, const block *ref_block,
                       block *next_block, int with_xor) {
    __m128i block_XY[ARGON2_OWORDS_IN_BLOCK];
    unsigned int i;

    if (with_xor) {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            state[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
            block_XY[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)next_block->v + i));
        }
    } else {
        for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
            block_XY[i] = state[i] = _mm_xor_si128(
                state[i], _mm_loadu_si128((const __m128i *)ref_block->v + i));
        }
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * i + 0], state[8 * i + 1], state[8 * i + 2],
            state[8 * i + 3], state[8 * i + 4], state[8 * i + 5],
            state[8 * i + 6], state[8 * i + 7]);
    }

    for (i = 0; i < 8; ++i) {
        BLAKE2_ROUND(state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i],
            state[8 * 3 + i], state[8 * 4 + i], state[8 * 5 + i],
            state[8 * 6 + i], state[8 * 7 + i]);
    }

    for (i = 0; i < ARGON2_OWORDS_IN_BLOCK; i++) {
        state[i] = _mm_xor_si128(state[i], block_XY[i]);
        _mm_storeu_si128((__m128i *)next_block->v + i, state[i]);
    }
}
#endif

static void next_addresses(block *address_block, block *input_block) {
    /*Temporary zero-initialized blocks*/
#if defined(__AVX512F__)
    __m512i zero_block[ARGON2_512BIT_WORDS_IN_BLOCK];
    __m512i zero2_block[ARGON2_512BIT_WORDS_IN_BLOCK];
#elif defined(__AVX2__)
    __m256i zero_block[ARGON2_HWORDS_IN_BLOCK];
    __m256i zero2_block[ARGON2_HWORDS_IN_BLOCK];
#else
    __m128i zero_block[ARGON2_OWORDS_IN_BLOCK];
    __m128i zero2_block[ARGON2_OWORDS_IN_BLOCK];
#endif

    memset(zero_block, 0, sizeof(zero_block));
    memset(zero2_block, 0, sizeof(zero2_block));

    /*Increasing index counter*/
    input_block->v[6]++;

    /*First iteration of G*/
    fill_block(zero_block, input_block, address_block, 0);

    /*Second iteration of G*/
    fill_block(zero2_block, address_block, address_block, 0);
}

uint32_t argon2i_index_size(const argon2_instance_t *instance) {
    return instance->segment_length * ARGON2_SYNC_POINTS - 2;
}

#define SIMPLE_PRECOMPUTE (0)

uint32_t argon2i_precompute(
    const argon2_instance_t *instance,
    argon2_precomputed_index_t *oIndex) {

    block address_block, input_block;
    uint64_t pseudo_rand;
    uint32_t starting_index, ref_index, slice;

    if (instance == NULL) {
        return 0;
    }

#define LOG_PRECOMPUTE (0)
#if LOG_PRECOMPUTE
    printf("-- argon2i_precompute %d,%d,%d\n", instance->lanes, instance->memory_blocks, instance->threads);
#endif

    // first compute normal ref block indices
    uint32_t nIndices = 0;
    for (slice = 0; slice < ARGON2_SYNC_POINTS; slice++) {
        init_block_value(&input_block, 0);
        input_block.v[0] = 0; // position.pass
        input_block.v[1] = 0; // position.lane
        input_block.v[2] = slice;
        input_block.v[3] = instance->memory_blocks;
        input_block.v[4] = instance->passes;
        input_block.v[5] = instance->type;

        if (slice == 0) {
            starting_index = 2;
            next_addresses(&address_block, &input_block);
        }
        else {
            starting_index = 0;
        }

        for (uint32_t i = starting_index; i < instance->segment_length; ++i) {
            if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
                next_addresses(&address_block, &input_block);
            }
            pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];

            argon2_position_t position = { 0, 0, (uint8_t)slice, i };
            ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF, 1);

            oIndex[nIndices++].refSlot = ref_index;
        }
    }

    size_t nBlocks;

#if SIMPLE_PRECOMPUTE
    nBlocks = instance->segment_length * ARGON2_SYNC_POINTS;
    if (nBlocks != (nIndices + 2)) {
        printf("\nfuck it\n");
    }
    for (uint32_t i = 2; i < nBlocks; i++) {
        oIndex[i - 2].store = 1;
        oIndex[i - 2].storeSlot = i;
    }
    return (uint32_t)nBlocks;
#endif

    // find step of last usage for each block
    nBlocks = instance->segment_length * ARGON2_SYNC_POINTS;
    uint32_t *stepLastUsage = malloc(nBlocks * sizeof(uint32_t));
    memset(stepLastUsage, UCHAR_MAX, nBlocks * sizeof(uint32_t));
    for (int i = nIndices - 1; i >= 0; i--) {
        int refIndex = oIndex[i].refSlot;
        if (stepLastUsage[refIndex] == UINT32_MAX) {
            stepLastUsage[refIndex] = i;
        }
    }
    
    // find blocks which are never used as ref
    for (uint32_t k = 0; k < nIndices; k++) {
        if (stepLastUsage[k] == UINT32_MAX) {
            stepLastUsage[k] = 0;
        }
    }

    // compute reindexing using a block pool
    size_t max_blocks_mem = nBlocks * sizeof(uint32_t);
    
    uint32_t* blockPool = malloc(max_blocks_mem);
    memset(blockPool, UCHAR_MAX, max_blocks_mem);
    
    uint32_t* blocksSlots = malloc(max_blocks_mem);
    memset(blocksSlots, UCHAR_MAX, max_blocks_mem);

    uint64_t storeCount = 0;
    uint32_t lastFreeSlot = 0;
    uint32_t blockPoolsize = 2;
    blockPool[0] = 0; blocksSlots[0] = 0;
    blockPool[1] = 1; blocksSlots[1] = 1;
    
    // for each step
    for (uint32_t step = 0; step < nIndices; step++) {
        
		uint32_t block = 2 + step;

        // every nth step do a cleanup of the pool
        const uint32_t CLEAN_RATE = 100;
        if ((step % CLEAN_RATE) == (CLEAN_RATE - 1)) {
            for (uint32_t j = 0; j < blockPoolsize; j++) {
                uint32_t poolBlock = blockPool[j];
                if (poolBlock != UINT32_MAX && step > stepLastUsage[poolBlock]) {
                    blockPool[j] = UINT32_MAX;
                    blocksSlots[poolBlock] = UINT32_MAX;
                }
            }
            lastFreeSlot = 0;
        }

		// store new block if needed
		uint32_t store = step <= stepLastUsage[block];
        argon2_precomputed_index_t* pIndex = oIndex + step;
		storeCount += store;

		uint32_t slot = UINT32_MAX;
		if (store) {
			// find empty slot in block pool
			for (uint32_t j = lastFreeSlot; j < blockPoolsize; j++) {
				uint32_t slotBlock = blockPool[j];
				if (slotBlock == UINT32_MAX) {
					lastFreeSlot = j;
					slot = j;
					break;
				}
			}
			// no empty slot found, increase pool size
			if (slot == UINT32_MAX) {
				slot = blockPoolsize++;
				lastFreeSlot = slot;
			}
			// store block in the pool slot
			blockPool[slot] = block;
			blocksSlots[block] = slot;
		}
//
#define TEST_REF_INDEX_IN_POOL (1)
#if TEST_REF_INDEX_IN_POOL
        uint32_t block_ref_index = pIndex->refSlot;
        const uint32_t TEST_RATE = 500;
        if ((step % TEST_RATE) == (TEST_RATE - 1)) {
            uint32_t new_ref_index = UINT32_MAX;
            for (uint32_t j = 0; j < blockPoolsize; j++) {
                if (blockPool[j] == block_ref_index) {
                    new_ref_index = j;
                    if (new_ref_index != blocksSlots[block_ref_index]) {
                        printf("TEST_PRECOMPUTE failed (1)\n");
                        exit(1);
                    }
                    break;
                }
            }
            if (new_ref_index == UINT32_MAX) {
                printf("TEST_PRECOMPUTE failed (2)\n");
                exit(1);
            }
        }
#endif

        // write final indexing values
        pIndex->store = store;
        pIndex->storeSlot = slot;
        pIndex->refSlot = blocksSlots[pIndex->refSlot];

#if LOG_PRECOMPUTE
        const uint32_t LOG_RATE = 100000;
        if ((step == (nIndices-1)) || (step % LOG_RATE) == (LOG_RATE - 1)) {
            printf("step=%8d  pool=%8u  => usage=%4.0f%% store=%4.0f%% progress=%4.0f%%\n",
                step + 1,
                blockPoolsize,
                100.0 * ((double)blockPoolsize / (double)nBlocks),
                100.0 * ((double)storeCount / (double)nIndices),
                100.0 * ((double)step / (nIndices-1.0)));
        }
#endif
    }

    // free memory
    free(blockPool);
    free(stepLastUsage);

    // return number of blocks actually needed
    return blockPoolsize;
}

void fill_memory_blocks_precompute(const argon2_instance_t *instance) {
#if defined(__AVX512F__)
    __m512i state[ARGON2_512BIT_WORDS_IN_BLOCK];
#elif defined(__AVX2__)
    __m256i state[ARGON2_HWORDS_IN_BLOCK];
#else
    __m128i state[ARGON2_OWORDS_IN_BLOCK];
#endif
    
    uint32_t nSteps = instance->segment_length * ARGON2_SYNC_POINTS;

    uint32_t starting_index = 2;
    uint32_t prev_offset = starting_index - 1;
    
    const argon2_precomputed_index_t* pIndex = instance->pPrecomputedIndex;

    memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

    for (uint32_t i = starting_index; i < nSteps; ++i) {
        block *ref_block = instance->memory + pIndex->refSlot;
        block *curr_block = instance->memory + pIndex->storeSlot;
        if (!pIndex->store)
            curr_block = NULL;
        fill_block_precompute(state, ref_block, curr_block);
        pIndex++;
    }

	memcpy(instance->memory + instance->memory_blocks - 1, state, ARGON2_BLOCK_SIZE);
}

#if SIMPLIFIED_FILL_SEGMENT
void fill_segment(const argon2_instance_t *instance, argon2_position_t position) {
	block *ref_block = NULL, *curr_block = NULL;
	block address_block, input_block;
	uint64_t pseudo_rand, ref_index;
	uint32_t prev_offset, curr_offset;
	uint32_t starting_index, i;
#if defined(__AVX512F__)
	__m512i state[ARGON2_512BIT_WORDS_IN_BLOCK];
#elif defined(__AVX2__)
	__m256i state[ARGON2_HWORDS_IN_BLOCK];
#else
	__m128i state[ARGON2_OWORDS_IN_BLOCK];
#endif
	if (instance == NULL) {
		return;
	}

    SPAM("fill_segment, pass=%u lane=%u slice=%u index=%u ",
		position.pass,
		position.lane,
		position.slice,
		position.index);

	// set input block to 0
	init_block_value(&input_block, 0);

	input_block.v[0] = 0; // position.pass
	input_block.v[1] = 0; // position.lane
	input_block.v[2] = position.slice;
	input_block.v[3] = instance->memory_blocks;
	input_block.v[4] = instance->passes;
	input_block.v[5] = instance->type;

	starting_index = 0;

	if (position.slice == 0) {
		/* we have already generated the first two blocks */
		starting_index = 2; 
		/* Don't forget to generate the first block of addresses: */
		next_addresses(&address_block, &input_block);
	}

	/* Offset of the current block */
	curr_offset = position.slice * instance->segment_length + starting_index;

	if (0 == curr_offset % instance->lane_length) {
		/* Last block in this lane */
		prev_offset = curr_offset + instance->lane_length - 1;
	}
	else {
		/* Previous block */
		prev_offset = curr_offset - 1;
	}

    SPAM("starting index = %u ", starting_index);
    SPAM("curr_offset = %u ", curr_offset);
    SPAM("prev_offset=%d ", prev_offset);
    SPAM("\n");

	memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

	for (i = starting_index; i < instance->segment_length;
		++i, ++curr_offset, ++prev_offset) {

		/* 1.2 Computing the index of the reference block */
		/* 1.2.1 Taking pseudo-random value from the previous block */
		if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
			next_addresses(&address_block, &input_block);
		}
		pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];

		/* 1.2.3 Computing the number of possible reference block within the
		* lane.
		*/
		position.index = i;
		ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF, 1);

		/* 2 Creating a new block */
		ref_block =
			instance->memory + ref_index;

		curr_block = instance->memory + curr_offset;

#if 0
        SPAM("%8lu / %8lu curr_offset=%4lu ref_index=%4lu\n",
			i,
			instance->segment_length,
			curr_offset,
			ref_index);
#endif

		fill_block(state, ref_block, curr_block, 0);
	}
}
#else
void fill_segment(const argon2_instance_t *instance,
                  argon2_position_t position) {
    block *ref_block = NULL, *curr_block = NULL;
    block address_block, input_block;
    uint64_t pseudo_rand, ref_index, ref_lane;
    uint32_t prev_offset, curr_offset;
    uint32_t starting_index, i;
#if defined(__AVX512F__)
    __m512i state[ARGON2_512BIT_WORDS_IN_BLOCK];
#elif defined(__AVX2__)
    __m256i state[ARGON2_HWORDS_IN_BLOCK];
#else
    __m128i state[ARGON2_OWORDS_IN_BLOCK];
#endif
    if (instance == NULL) {
        return;
    }

    init_block_value(&input_block, 0);

    input_block.v[0] = position.pass;
    input_block.v[1] = position.lane;
    input_block.v[2] = position.slice;
    input_block.v[3] = instance->memory_blocks;
    input_block.v[4] = instance->passes;
    input_block.v[5] = instance->type;

    starting_index = 0;

    if ((0 == position.pass) && (0 == position.slice)) {
        starting_index = 2; /* we have already generated the first two blocks */

        /* Don't forget to generate the first block of addresses: */
        next_addresses(&address_block, &input_block);
    }

    /* Offset of the current block */
    curr_offset = position.lane * instance->lane_length +
                  position.slice * instance->segment_length + starting_index;

    if (0 == curr_offset % instance->lane_length) {
        /* Last block in this lane */
        prev_offset = curr_offset + instance->lane_length - 1;
    } else {
        /* Previous block */
        prev_offset = curr_offset - 1;
    }

    memcpy(state, ((instance->memory + prev_offset)->v), ARGON2_BLOCK_SIZE);

    for (i = starting_index; i < instance->segment_length;
         ++i, ++curr_offset, ++prev_offset) {
        /*1.1 Rotating prev_offset if needed */
        if (curr_offset % instance->lane_length == 1) {
            prev_offset = curr_offset - 1;
        }

        /* 1.2 Computing the index of the reference block */
        /* 1.2.1 Taking pseudo-random value from the previous block */
        if (i % ARGON2_ADDRESSES_IN_BLOCK == 0) {
            next_addresses(&address_block, &input_block);
        }
        pseudo_rand = address_block.v[i % ARGON2_ADDRESSES_IN_BLOCK];

        /* 1.2.2 Computing the lane of the reference block */
        ref_lane = ((pseudo_rand >> 32)) % instance->lanes;

        if ((position.pass == 0) && (position.slice == 0)) {
            /* Can not reference other lanes yet */
            ref_lane = position.lane;
        }

        /* 1.2.3 Computing the number of possible reference block within the
         * lane.
         */
        position.index = i;
        ref_index = index_alpha(instance, &position, pseudo_rand & 0xFFFFFFFF,
                                ref_lane == position.lane);

        /* 2 Creating a new block */
        ref_block =
            instance->memory + instance->lane_length * ref_lane + ref_index;
        curr_block = instance->memory + curr_offset;
        if(0 == position.pass) {
            fill_block(state, ref_block, curr_block, 0);
        } else {
            fill_block(state, ref_block, curr_block, 1);
        }
    }
}
#endif