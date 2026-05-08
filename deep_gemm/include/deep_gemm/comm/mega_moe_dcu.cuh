#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>

#include <deep_gemm/layout/mega_moe_dcu.cuh>

namespace deep_gemm::mega {

static constexpr int kBarrierFinishedTag = 1024;
static constexpr uint64_t kFullWaveMask = ~0ull;
static constexpr long long kBarrierTimeoutCycles = 60000000000ll;

struct SymBufferSections {
    const uint8_t* x;
    const float* x_sf;
    const int64_t* topk_idx;
    const float* topk_weights;
    uint16_t* combine;
};

__device__ static inline SymBufferSections get_sections(uint8_t* sym_buffer,
                                                        const int num_ranks,
                                                        const int num_experts,
                                                        const int num_max_tokens_per_rank,
                                                        const int num_topk,
                                                        const int hidden) {
    const int64_t input_token_offset = workspace_bytes(num_ranks, num_experts, num_max_tokens_per_rank);
    const int64_t input_sf_offset =
        input_token_offset + static_cast<int64_t>(num_max_tokens_per_rank) * hidden;
    const int64_t topk_idx_offset =
        input_sf_offset + static_cast<int64_t>(num_max_tokens_per_rank) * sizeof(float);
    const int64_t topk_weights_offset =
        topk_idx_offset + static_cast<int64_t>(num_max_tokens_per_rank) * num_topk * sizeof(int64_t);
    const int64_t combine_offset =
        topk_weights_offset + static_cast<int64_t>(num_max_tokens_per_rank) * num_topk * sizeof(float);
    return {
        sym_buffer + input_token_offset,
        reinterpret_cast<const float*>(sym_buffer + input_sf_offset),
        reinterpret_cast<const int64_t*>(sym_buffer + topk_idx_offset),
        reinterpret_cast<const float*>(sym_buffer + topk_weights_offset),
        reinterpret_cast<uint16_t*>(sym_buffer + combine_offset),
    };
}

__device__ static inline int load_signal_system(const volatile int* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ static inline int wave_all_sync(const uint64_t mask, const int predicate) {
    const uint64_t predicate_bits = __ballot(predicate);
    return (~predicate_bits & mask) == 0;
}

__device__ static inline void mega_moe_rank_barrier(int** signal_buffers,
                                                    const int rank_idx,
                                                    const int num_ranks) {
    const int thread_id = static_cast<int>(threadIdx.x);
    auto* my_signals = signal_buffers[rank_idx];
    __threadfence_system();
    __syncthreads();

    if (thread_id < num_ranks) {
        auto* peer_signals = signal_buffers[thread_id];
        atomicAdd_system(my_signals + thread_id, kBarrierFinishedTag);
        atomicSub_system(peer_signals + rank_idx, kBarrierFinishedTag);
    }

    const auto start_time = clock64();
    while (true) {
        int value = 0;
        if (thread_id < num_ranks) {
            volatile int* signal = reinterpret_cast<volatile int*>(my_signals + thread_id);
            value = load_signal_system(signal);
        }
        if (wave_all_sync(kFullWaveMask, value <= 0))
            break;
        if (clock64() - start_time > kBarrierTimeoutCycles && thread_id < num_ranks) {
            printf("MegaMoE HIP rank barrier timeout: rank=%d thread=%d value=%d\n",
                   rank_idx, thread_id, value);
            abort();
        }
    }
    __syncthreads();
}

__device__ static inline void mega_moe_local_blocks_barrier(int* signals,
                                                            const int rank_idx,
                                                            const int num_blocks) {
    constexpr int kLocalBarrierCountSlot = 16;
    constexpr int kLocalBarrierSenseSlot = 17;
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile int* sense_ptr = reinterpret_cast<volatile int*>(signals + kLocalBarrierSenseSlot);
        const int sense = load_signal_system(sense_ptr);
        const int old = atomicAdd_system(signals + kLocalBarrierCountSlot, 1);
        if (old == num_blocks - 1) {
            __threadfence_system();
            atomicSub_system(signals + kLocalBarrierCountSlot, num_blocks);
            atomicAdd_system(signals + kLocalBarrierSenseSlot, 1);
        } else {
            const auto start_time = clock64();
            while (load_signal_system(sense_ptr) == sense) {
                if (clock64() - start_time > kBarrierTimeoutCycles) {
                    printf("MegaMoE HIP local barrier timeout: rank=%d block=%d sense=%d\n",
                           rank_idx, static_cast<int>(blockIdx.x), sense);
                    abort();
                }
            }
        }
    }
    __syncthreads();
}

} // namespace deep_gemm::mega
