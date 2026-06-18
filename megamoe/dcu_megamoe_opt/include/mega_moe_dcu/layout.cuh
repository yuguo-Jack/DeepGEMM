#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace deep_gemm::mega {

static constexpr int kTokenAlignment = 384;

__host__ __device__ static inline int64_t align_i64(const int64_t value, const int64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__host__ __device__ static inline int64_t dcu_sym_buffer_ptrs_offset() {
    return 0;
}

__host__ __device__ static inline int64_t dcu_signal_ptrs_offset(const int num_ranks) {
    return align_i64(static_cast<int64_t>(num_ranks) * sizeof(uint8_t*), 16);
}

__host__ __device__ static inline int64_t dcu_workspace_offset(const int num_ranks) {
    return align_i64(
        dcu_signal_ptrs_offset(num_ranks) +
            static_cast<int64_t>(num_ranks) * sizeof(int*),
        16);
}

__host__ __device__ static inline int64_t dcu_runtime_num_tokens_offset(const int num_ranks) {
    return dcu_workspace_offset(num_ranks);
}

__host__ __device__ static inline int64_t dcu_uniform_num_tokens_offset(const int num_ranks) {
    return dcu_runtime_num_tokens_offset(num_ranks) + sizeof(int32_t);
}

__host__ __device__ static inline int64_t dcu_input_token_offset(const int num_ranks) {
    return align_i64(dcu_uniform_num_tokens_offset(num_ranks) + sizeof(int32_t), 16);
}

__host__ __device__ static inline int64_t marlin_nt_kpack2_offset(
    const int expert_idx,
    const int row_idx,
    const int k_idx,
    const int rows,
    const int k) {
    constexpr int kTileN = 16;
    constexpr int kTileK = 16;
    const int row_tile = row_idx / kTileN;
    const int row_inner = row_idx - row_tile * kTileN;
    const int k_tile = k_idx / kTileK;
    const int k_inner = k_idx - k_tile * kTileK;
    const int row_tiles = rows / kTileN;
    return ((static_cast<int64_t>(expert_idx) * row_tiles + row_tile) * k +
            k_tile * kTileN) * kTileK +
           row_inner * kTileK + k_inner;
}

__host__ __device__ static inline int64_t marlin_nt_kpack2_row_base_offset(
    const int expert_idx,
    const int row_idx,
    const int rows,
    const int k) {
    constexpr int kTileN = 16;
    constexpr int kTileK = 16;
    const int row_tile = row_idx / kTileN;
    const int row_inner = row_idx - row_tile * kTileN;
    const int row_tiles = rows / kTileN;
    return (static_cast<int64_t>(expert_idx) * row_tiles + row_tile) * k * kTileK +
           row_inner * kTileK;
}

__host__ __device__ static inline int marlin_nt_kpack2_k_offset(const int k_idx) {
    constexpr int kTileN = 16;
    constexpr int kTileK = 16;
    const int k_tile = k_idx / kTileK;
    const int k_inner = k_idx - k_tile * kTileK;
    return k_tile * kTileN * kTileK + k_inner;
}

__host__ __device__ static inline int64_t workspace_task_capacity_per_expert(
    const int num_ranks,
    const int num_max_tokens_per_rank) {
    return static_cast<int64_t>(num_ranks) * num_ranks * num_max_tokens_per_rank;
}

__host__ __device__ static inline int64_t workspace_bytes(const int num_ranks,
                                                          const int,
                                                          const int) {
    return dcu_input_token_offset(num_ranks);
}

__host__ __device__ static inline int64_t route_task_workspace_bytes(
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank) {
    const int num_experts_per_rank = num_experts / num_ranks;
    int64_t bytes = 0;
    bytes += static_cast<int64_t>(num_experts_per_rank) * sizeof(int);
    bytes = align_i64(bytes, 16);
    bytes += static_cast<int64_t>(num_experts_per_rank) *
             workspace_task_capacity_per_expert(num_ranks, num_max_tokens_per_rank) *
             sizeof(int);
    return align_i64(bytes, 16);
}

__host__ __device__ static inline int64_t combine_token_offset(const int num_ranks,
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
    return topk_weights_offset +
           static_cast<int64_t>(num_max_tokens_per_rank) * num_topk * sizeof(float);
}

__host__ __device__ static inline uint8_t** dcu_peer_sym_buffer_ptrs(uint8_t* sym_buffer) {
    return reinterpret_cast<uint8_t**>(sym_buffer + dcu_sym_buffer_ptrs_offset());
}

__host__ __device__ static inline int** dcu_peer_signal_ptrs(uint8_t* sym_buffer,
                                                            const int num_ranks) {
    return reinterpret_cast<int**>(sym_buffer + dcu_signal_ptrs_offset(num_ranks));
}

__host__ __device__ static inline int32_t* dcu_runtime_num_tokens_ptr(uint8_t* sym_buffer,
                                                                      const int num_ranks) {
    return reinterpret_cast<int32_t*>(sym_buffer + dcu_runtime_num_tokens_offset(num_ranks));
}

__host__ __device__ static inline int32_t* dcu_uniform_num_tokens_ptr(uint8_t* sym_buffer,
                                                                      const int num_ranks) {
    return reinterpret_cast<int32_t*>(sym_buffer + dcu_uniform_num_tokens_offset(num_ranks));
}

} // namespace deep_gemm::mega
