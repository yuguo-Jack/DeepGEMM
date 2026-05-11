#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace deep_gemm::mega {

static constexpr int kTokenAlignment = 384;
static constexpr int kDcuRouteTileMMinLog2 = 4;
static constexpr int kDcuRouteTileMDefaultLog2 = 5;
static constexpr int kDcuRouteScratchMmacTileM = 16;
static constexpr int kDcuRouteScratchPipelineCounterCount = 5;

__host__ __device__ static inline int64_t align_i64(const int64_t value, const int64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

__host__ __device__ static inline int dcu_route_tile_m_from_log2(const int tile_m_log2) {
    return 1 << tile_m_log2;
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
    return dcu_workspace_offset(num_ranks);
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

__host__ __device__ static inline int64_t dcu_route_scratch_capacity_tiles(
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank,
    const int num_topk,
    const int tile_m) {
    const int num_experts_per_rank = num_experts / num_ranks;
    const int64_t max_route_tasks =
        static_cast<int64_t>(num_ranks) * num_max_tokens_per_rank * num_topk;
    return num_experts_per_rank + (max_route_tasks + tile_m - 1) / tile_m;
}

struct DcuRouteTileScratchLayout {
    int tile_m;
    int64_t x_fp8_offset;
    int64_t act_bf16_offset;
    int64_t act_fp8_offset;
    int64_t act_scale_offset;
    int64_t act_chunk_amax_offset;
    int64_t tile_x_row_ptrs_offset;
    int64_t tile_combine_row_ptrs_offset;
    int64_t tile_route_weight_offset;
    int64_t tile_x_scale_offset;
    int64_t tile_expert_offset;
    int64_t tile_pool_base_offset;
    int64_t tile_count_offset;
    int64_t expert_l1_task_offset;
    int64_t expert_quant_done_count_offset;
    int64_t l2_group_done_count_offset;
    int64_t tile_pull_done_offset;
    int64_t l1_done_count_offset;
    int64_t l2_queue_offset;
    int64_t l2_queue_ready_offset;
    int64_t pipeline_counter_offset;
    int64_t total_tiles_offset;
    int64_t bytes;
};

__host__ __device__ static inline DcuRouteTileScratchLayout dcu_route_tile_scratch_layout(
    const int64_t max_route_tiles,
    const int tile_m,
    const int hidden,
    const int intermediate_hidden) {
    DcuRouteTileScratchLayout layout{};
    layout.tile_m = tile_m;
    int64_t offset = 0;
    layout.x_fp8_offset = offset;
    offset += max_route_tiles * tile_m * static_cast<int64_t>(hidden);
    offset = align_i64(offset, 16);
    layout.act_bf16_offset = offset;
    offset += max_route_tiles * tile_m * intermediate_hidden * static_cast<int64_t>(sizeof(uint16_t));
    offset = align_i64(offset, 16);
    layout.act_fp8_offset = offset;
    offset += max_route_tiles * tile_m * intermediate_hidden;
    offset = align_i64(offset, 16);
    layout.act_scale_offset = offset;
    offset += max_route_tiles * tile_m * static_cast<int64_t>(sizeof(float));
    offset = align_i64(offset, 16);
    layout.act_chunk_amax_offset = offset;
    offset += max_route_tiles * tile_m *
              ((intermediate_hidden + 255) / 256) *
              static_cast<int64_t>(sizeof(float));
    offset = align_i64(offset, 16);
    layout.tile_x_row_ptrs_offset = offset;
    offset += max_route_tiles * tile_m * static_cast<int64_t>(sizeof(uint8_t*));
    offset = align_i64(offset, 16);
    layout.tile_combine_row_ptrs_offset = offset;
    offset += max_route_tiles * tile_m * static_cast<int64_t>(sizeof(uint16_t*));
    offset = align_i64(offset, 16);
    layout.tile_route_weight_offset = offset;
    offset += max_route_tiles * tile_m * static_cast<int64_t>(sizeof(float));
    offset = align_i64(offset, 16);
    layout.tile_x_scale_offset = offset;
    offset += max_route_tiles * tile_m * static_cast<int64_t>(sizeof(float));
    offset = align_i64(offset, 16);
    layout.tile_expert_offset = offset;
    offset += max_route_tiles * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.tile_pool_base_offset = offset;
    offset += max_route_tiles * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.tile_count_offset = offset;
    offset += max_route_tiles * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.expert_l1_task_offset = offset;
    offset += (max_route_tiles + 1) * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.expert_quant_done_count_offset = offset;
    offset += (max_route_tiles + 1) * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    const int subtiles_per_tile = tile_m / kDcuRouteScratchMmacTileM;
    const int64_t max_subtiles = max_route_tiles * subtiles_per_tile;
    layout.l2_group_done_count_offset = offset;
    offset += max_subtiles * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.tile_pull_done_offset = offset;
    offset += max_route_tiles * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.l1_done_count_offset = offset;
    offset += max_subtiles * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    const int64_t max_l2_tasks = max_subtiles * ((hidden + 15) / 16);
    layout.l2_queue_offset = offset;
    offset += max_l2_tasks * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.l2_queue_ready_offset = offset;
    offset += max_l2_tasks * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.pipeline_counter_offset = offset;
    offset += kDcuRouteScratchPipelineCounterCount * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.total_tiles_offset = offset;
    offset += static_cast<int64_t>(sizeof(int));
    layout.bytes = align_i64(offset, 16);
    return layout;
}

__host__ __device__ static inline int64_t dcu_route_scratch_bytes(
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank,
    const int num_topk,
    const int tile_m,
    const int hidden,
    const int intermediate_hidden) {
    const int64_t route_scratch_tiles = dcu_route_scratch_capacity_tiles(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, tile_m);
    const auto route_tile_layout = dcu_route_tile_scratch_layout(
        route_scratch_tiles, tile_m, hidden, intermediate_hidden);
    return align_i64(
        route_task_workspace_bytes(num_ranks, num_experts, num_max_tokens_per_rank) +
            route_tile_layout.bytes,
        16);
}

__host__ __device__ static inline uint8_t** dcu_peer_sym_buffer_ptrs(uint8_t* sym_buffer) {
    return reinterpret_cast<uint8_t**>(sym_buffer + dcu_sym_buffer_ptrs_offset());
}

__host__ __device__ static inline int** dcu_peer_signal_ptrs(uint8_t* sym_buffer,
                                                            const int num_ranks) {
    return reinterpret_cast<int**>(sym_buffer + dcu_signal_ptrs_offset(num_ranks));
}

__device__ static inline int* route_scratch_expert_counts(uint8_t* route_scratch) {
    return reinterpret_cast<int*>(route_scratch);
}

__device__ static inline int* route_scratch_expert_task_pool(
    uint8_t* route_scratch,
    const int num_experts_per_rank) {
    const int64_t offset = align_i64(
        static_cast<int64_t>(num_experts_per_rank) * sizeof(int), 16);
    return reinterpret_cast<int*>(route_scratch + offset);
}

__host__ __device__ static inline uint8_t* route_tile_scratch_base(
    uint8_t* route_scratch,
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank) {
    return route_scratch +
           route_task_workspace_bytes(num_ranks, num_experts, num_max_tokens_per_rank);
}

} // namespace deep_gemm::mega
