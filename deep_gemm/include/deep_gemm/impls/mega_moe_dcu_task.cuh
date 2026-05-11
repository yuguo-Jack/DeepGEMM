#pragma once

#include <deep_gemm/common/mega_moe_dcu.cuh>

namespace deep_gemm::mega {

static constexpr int kDcuRouteTileMMinLog2 = 4;
static constexpr int kDcuPipelinePullTileHead = 0;
static constexpr int kDcuPipelineL1TaskHead = 1;
static constexpr int kDcuPipelineL2QueueTail = 2;
static constexpr int kDcuPipelineL2QueueHead = 3;
static constexpr int kDcuPipelineL2Done = 4;
static constexpr int kDcuPipelineCounterCount = 5;

__host__ __device__ static inline int dcu_route_tile_m_from_log2(const int tile_m_log2) {
    return 1 << tile_m_log2;
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
    const int subtiles_per_tile = tile_m / kDcuMmacTileM;
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
    offset += kDcuPipelineCounterCount * static_cast<int64_t>(sizeof(int));
    offset = align_i64(offset, 16);
    layout.total_tiles_offset = offset;
    offset += static_cast<int64_t>(sizeof(int));
    layout.bytes = align_i64(offset, 16);
    return layout;
}

__device__ static inline int dcu_load_acquire_int(const volatile int* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ static inline int dcu_fetch_add_release_int(int* ptr, const int value) {
    return __hip_atomic_fetch_add(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ static inline void dcu_store_release_int(int* ptr, const int value) {
    __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ static inline int dcu_route_subtile_valid_rows(
    const int tile_count,
    const int subtile_idx) {
    const int first_row = subtile_idx * kDcuMmacTileM;
    const int remaining = tile_count - first_row;
    if (remaining <= 0)
        return 0;
    return remaining < kDcuMmacTileM ? remaining : kDcuMmacTileM;
}

__device__ static inline int dcu_route_mtile_valid_rows(
    const int tile_count,
    const int mtile_row_start,
    const int mtile_m) {
    const int remaining = tile_count - mtile_row_start;
    if (remaining <= 0)
        return 0;
    return remaining < mtile_m ? remaining : mtile_m;
}

__device__ static inline int64_t dcu_route_meta_base(
    const int tile_id,
    const int tile_m_log2,
    const int subtile_idx) {
    return (static_cast<int64_t>(tile_id) << tile_m_log2) +
           (static_cast<int64_t>(subtile_idx) << kDcuMmacTileMLog2);
}

__device__ static inline void prepare_dcu_route_tile_metadata(
    uint8_t** sym_buffers,
    const int* expert_task_pool,
    const int64_t max_tasks_per_expert,
    const int total_tiles,
    const int tile_m_log2,
    const int* tile_experts,
    const int* tile_pool_bases,
    const int* tile_counts,
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank,
    const int num_tokens,
    const int num_topk,
    const int hidden,
    const int block_stride,
    const int thread_offset,
    const uint8_t** tile_x_row_ptrs,
    uint16_t** tile_combine_row_ptrs,
    float* tile_route_weights,
    float* tile_x_scales) {
    const int tile_m = dcu_route_tile_m_from_log2(tile_m_log2);
    const int tile_m_mask = tile_m - 1;
    const int64_t total_rows = static_cast<int64_t>(total_tiles) * tile_m;
    for (int64_t linear_row = thread_offset;
         linear_row < total_rows;
         linear_row += block_stride) {
        const int tile_id = static_cast<int>(linear_row >> tile_m_log2);
        const int row = static_cast<int>(linear_row) & tile_m_mask;
        const int local_expert = tile_experts[tile_id];
        const int pool_idx = tile_pool_bases[tile_id] + row;
        const bool row_valid = row < tile_counts[tile_id];

        const int64_t meta_idx = (static_cast<int64_t>(tile_id) << tile_m_log2) + row;
        const uint8_t* x_row = nullptr;
        uint16_t* combine_row = nullptr;
        float route_weight = 0.0f;
        float x_scale = 1.0f;

        if (row_valid) {
            const int task =
                expert_task_pool[static_cast<int64_t>(local_expert) * max_tasks_per_expert + pool_idx];
            const int topk_slot = task % num_topk;
            const int token_idx = static_cast<int>((task / num_topk) % num_tokens);
            const int source_rank =
                static_cast<int>(task / (static_cast<int64_t>(num_topk) * num_tokens));
            auto sections = get_sections(
                sym_buffers[source_rank], num_ranks, num_experts,
                num_max_tokens_per_rank, num_topk, hidden);
            const int64_t route_offset = static_cast<int64_t>(token_idx) * num_topk + topk_slot;
            x_row = sections.x + static_cast<int64_t>(token_idx) * hidden;
            combine_row = sections.combine +
                (static_cast<int64_t>(topk_slot) * num_max_tokens_per_rank + token_idx) * hidden;
            route_weight = sections.topk_weights[route_offset];
            x_scale = sections.x_sf[token_idx];
        }

        tile_x_row_ptrs[meta_idx] = x_row;
        tile_combine_row_ptrs[meta_idx] = combine_row;
        tile_route_weights[meta_idx] = route_weight;
        tile_x_scales[meta_idx] = x_scale;
    }
}

__device__ static inline void pull_one_dcu_route_tile_x_pool(
    const uint8_t* const* tile_x_row_ptrs,
    const int* tile_counts,
    const int tile_id,
    const int tile_m_log2,
    const int hidden,
    const int block_stride,
    const int thread_offset,
    uint8_t* tile_x_pool) {
    constexpr int kVecBytes = 16;
    const int tile_m = dcu_route_tile_m_from_log2(tile_m_log2);
    const int vecs_per_row = hidden / kVecBytes;
    const int64_t total_vecs = static_cast<int64_t>(tile_m) * vecs_per_row;
    auto* dst_vecs = reinterpret_cast<uint4*>(
        tile_x_pool + static_cast<int64_t>(tile_id) * tile_m * hidden);

    for (int64_t linear_vec = thread_offset;
         linear_vec < total_vecs;
         linear_vec += block_stride) {
        const int row_linear = static_cast<int>(linear_vec / vecs_per_row);
        const int vec_idx =
            static_cast<int>(linear_vec - static_cast<int64_t>(row_linear) * vecs_per_row);
        const int row = row_linear;
        const int64_t meta_idx = (static_cast<int64_t>(tile_id) << tile_m_log2) + row;
        uint4 value{0, 0, 0, 0};
        if (row < tile_counts[tile_id]) {
            const auto* src_vecs = reinterpret_cast<const uint4*>(tile_x_row_ptrs[meta_idx]);
            value = src_vecs[vec_idx];
        }
        dst_vecs[linear_vec] = value;
    }
}

} // namespace deep_gemm::mega
