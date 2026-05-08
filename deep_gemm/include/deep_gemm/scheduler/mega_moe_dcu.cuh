#pragma once

#include <deep_gemm/impls/mega_moe_dcu_task.cuh>
#include <deep_gemm/impls/mega_moe_dcu_tiles.cuh>

namespace deep_gemm::mega {

__device__ static inline bool dcu_try_run_l2_queue_task(
    int* block_l2_queue_idx,
    int* pipeline_counters,
    const int* l2_queue,
    int* l2_queue_ready,
    const int num_hidden_chunks,
    const int subtiles_per_tile,
    const int* tile_counts,
    const int* tile_experts,
    const int route_tile_log2_m,
    const int hidden_chunk,
    const int hidden,
    const int intermediate_hidden,
    uint16_t* const* tile_combine_row_ptrs,
    const uint8_t* l2_weights,
    const float* l2_weights_sf,
    const uint8_t* all_act_fp8,
    const float* all_act_scale) {
    if (threadIdx.x == 0) {
        *block_l2_queue_idx = -1;
        while (true) {
            const int head = dcu_load_acquire_int(
                reinterpret_cast<volatile int*>(pipeline_counters + 2));
            const int tail = dcu_load_acquire_int(
                reinterpret_cast<volatile int*>(pipeline_counters + 1));
            if (head >= tail)
                break;
            if (atomicCAS(pipeline_counters + 2, head, head + 1) == head) {
                *block_l2_queue_idx = head;
                break;
            }
        }
    }
    __syncthreads();

    const bool claimed = *block_l2_queue_idx >= 0;
    if (claimed) {
        while (dcu_load_acquire_int(
                   reinterpret_cast<volatile int*>(l2_queue_ready + *block_l2_queue_idx)) == 0) {}
        const int task_id = l2_queue[*block_l2_queue_idx];
        const int subtile_task_id = task_id / num_hidden_chunks;
        const int chunk_id = task_id - subtile_task_id * num_hidden_chunks;
        const int tile_id = subtile_task_id / subtiles_per_tile;
        const int subtile_idx = subtile_task_id - tile_id * subtiles_per_tile;
        const int valid_rows = dcu_route_subtile_valid_rows(tile_counts[tile_id], subtile_idx);
        if (valid_rows > 0) {
            const int64_t tile_meta_base = dcu_route_meta_base(tile_id, route_tile_log2_m, subtile_idx);
            const int64_t act_offset =
                ((static_cast<int64_t>(tile_id) << route_tile_log2_m) +
                 (static_cast<int64_t>(subtile_idx) << kDcuMmacTileMLog2)) * intermediate_hidden;
            const int64_t scale_offset =
                (static_cast<int64_t>(tile_id) << route_tile_log2_m) +
                (static_cast<int64_t>(subtile_idx) << kDcuMmacTileMLog2);
            compute_route_mmac_mtile16_l2_chunk(
                tile_experts[tile_id], chunk_id * hidden_chunk,
                valid_rows,
                tile_meta_base,
                hidden, intermediate_hidden,
                tile_combine_row_ptrs,
                l2_weights, l2_weights_sf,
                all_act_fp8 + act_offset,
                all_act_scale + scale_offset);
        }
        if (threadIdx.x == 0)
            dcu_fetch_add_release_int(pipeline_counters + 3, 1);
    }
    __syncthreads();
    return claimed;
}

} // namespace deep_gemm::mega
