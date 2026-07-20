#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace deep_gemm::mega {

static constexpr int kTokenAlignment = 384;
static constexpr int kDcuMegaMoeStagedTopk = 6;
static constexpr int kDcuMegaMoeFlashExperts = 256;
static constexpr int kDcuMegaMoeFlashHidden = 4096;
static constexpr int kDcuMegaMoeFlashIntermediate = 2048;
static constexpr int kDcuMegaMoeProExperts = 384;
static constexpr int kDcuMegaMoeProHidden = 7168;
static constexpr int kDcuMegaMoeProIntermediate = 3072;
static constexpr int kDcuMegaMoeYgzpExperts = 288;
static constexpr int kDcuMegaMoeYgzpTopk = 8;
static constexpr int kDcuMegaMoeYgzpHidden = 4096;
static constexpr int kDcuMegaMoeYgzpIntermediate = 2048;
static constexpr int kDcuMegaMoeTailDoneCounterRingSlots = 16;
static constexpr int kDcuMegaMoeTailCopyExpertDoneCount = 64;
static constexpr int kDcuMegaMoeTailDoneCounterInts =
    3 * kDcuMegaMoeTailDoneCounterRingSlots + kDcuMegaMoeTailCopyExpertDoneCount;

__host__ __device__ static inline int64_t align_i64(const int64_t value, const int64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__host__ __device__ static inline bool dcu_supported_staged_ep_rank_count(const int num_ranks) {
    return num_ranks == 8 || num_ranks == 16 || num_ranks == 32;
}

__host__ __device__ static inline bool dcu_supported_staged_local_experts(
    const int local_experts) {
    return local_experts == 8 || local_experts == 12 || local_experts == 16 ||
           local_experts == 24 || local_experts == 32 || local_experts == 48;
}

__host__ __device__ static inline bool dcu_supported_staged_model_shape(
    const int num_experts,
    const int num_topk,
    const int hidden,
    const int intermediate_hidden) {
    return num_topk == kDcuMegaMoeStagedTopk &&
           ((num_experts == kDcuMegaMoeFlashExperts &&
             hidden == kDcuMegaMoeFlashHidden &&
             intermediate_hidden == kDcuMegaMoeFlashIntermediate) ||
            (num_experts == kDcuMegaMoeProExperts &&
             hidden == kDcuMegaMoeProHidden &&
             intermediate_hidden == kDcuMegaMoeProIntermediate));
}

__host__ __device__ static inline bool dcu_supported_staged_pack5_shape(
    const int num_ranks,
    const int num_experts,
    const int num_topk,
    const int hidden,
    const int intermediate_hidden) {
    return dcu_supported_staged_ep_rank_count(num_ranks) &&
           num_experts % num_ranks == 0 &&
           dcu_supported_staged_model_shape(
               num_experts, num_topk, hidden, intermediate_hidden);
}

__host__ __device__ static inline bool dcu_supported_staged_k1_shape(
    const int num_ranks,
    const int num_experts,
    const int num_topk,
    const int hidden,
    const int l1_rows) {
    return l1_rows % 2 == 0 &&
           dcu_supported_staged_pack5_shape(
               num_ranks, num_experts, num_topk, hidden, l1_rows / 2);
}

__host__ __device__ static inline bool dcu_supported_staged_k3_dims(
    const int hidden,
    const int intermediate_hidden) {
    return (hidden == kDcuMegaMoeFlashHidden &&
            intermediate_hidden == kDcuMegaMoeFlashIntermediate) ||
           (hidden == kDcuMegaMoeProHidden &&
           intermediate_hidden == kDcuMegaMoeProIntermediate);
}

__host__ __device__ static inline bool dcu_supported_staged_int8_normal_shape(
    const int num_ranks,
    const int num_experts,
    const int num_topk,
    const int hidden,
    const int intermediate_hidden) {
    return num_ranks == 8 &&
           num_experts == kDcuMegaMoeYgzpExperts &&
           num_topk == kDcuMegaMoeYgzpTopk &&
           hidden == kDcuMegaMoeYgzpHidden &&
           intermediate_hidden == kDcuMegaMoeYgzpIntermediate &&
           num_experts % num_ranks == 0;
}

__host__ __device__ static inline bool dcu_supported_staged_int8_normal_k1_shape(
    const int num_ranks,
    const int num_experts,
    const int num_topk,
    const int hidden,
    const int l1_rows) {
    return l1_rows == 2 * kDcuMegaMoeYgzpIntermediate &&
           dcu_supported_staged_int8_normal_shape(
               num_ranks, num_experts, num_topk, hidden,
               kDcuMegaMoeYgzpIntermediate);
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

__host__ __device__ static inline int dcu_tail_signal_slot_base(const int num_ranks) {
    return num_ranks <= 8 ? 8 : num_ranks;
}

__host__ __device__ static inline int dcu_start_barrier_signal_slot_base(const int num_ranks) {
    return num_ranks <= 8 ? 18 : dcu_tail_signal_slot_base(num_ranks) + num_ranks;
}

__host__ __device__ static inline int dcu_post_k3_barrier_signal_slot_base(const int num_ranks) {
    return num_ranks <= 8 ? 20 : dcu_start_barrier_signal_slot_base(num_ranks) + 2;
}

__host__ __device__ static inline int dcu_split_tail_chunk_signal_slot_base(const int num_ranks) {
    return num_ranks <= 8 ? 22 : dcu_post_k3_barrier_signal_slot_base(num_ranks) + 2;
}

__host__ __device__ static inline int dcu_required_signal_slots(const int num_ranks) {
    return dcu_split_tail_chunk_signal_slot_base(num_ranks) + 8;
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
