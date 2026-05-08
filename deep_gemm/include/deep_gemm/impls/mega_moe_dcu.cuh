#pragma once

#include <deep_gemm/common/mega_moe_dcu.cuh>
#include <deep_gemm/comm/mega_moe_dcu.cuh>
#include <deep_gemm/impls/mega_moe_dcu_task.cuh>
#include <deep_gemm/impls/mega_moe_dcu_tiles.cuh>

namespace deep_gemm::mega {

template <int kStaticValue>
__device__ static inline int dcu_static_or_runtime(const int runtime_value) {
    if constexpr (kStaticValue > 0)
        return kStaticValue;
    return runtime_value;
}

template <
    int kRouteTileLog2M,
    int kStaticNumRanks = 0,
    int kStaticTopK = 0,
    int kStaticHidden = 0,
    int kStaticIntermediate = 0,
    int kStaticExpertsPerRank = 0>
__global__ __launch_bounds__(512) void mega_moe_multirank_persistent_w8a8_channelwise_kernel(
    uint16_t* y,
    uint8_t** sym_buffers,
    int** signal_buffers,
    uint8_t* route_scratch,
    const int64_t route_scratch_tiles,
    const uint8_t* l1_weights, const float* l1_weights_sf,
    const uint8_t* l2_weights, const float* l2_weights_sf,
    int* cumulative_local_expert_recv_stats,
    const int rank_idx, const int num_ranks,
    const int num_max_tokens_per_rank,
    const int num_experts_per_rank,
    const int num_tokens, const int num_topk,
    const int hidden_runtime, const int intermediate_hidden_runtime,
    const float activation_clamp) {
    const int num_ranks_static = dcu_static_or_runtime<kStaticNumRanks>(num_ranks);
    const int num_experts_per_rank_static =
        dcu_static_or_runtime<kStaticExpertsPerRank>(num_experts_per_rank);
    const int num_topk_static = dcu_static_or_runtime<kStaticTopK>(num_topk);
    const int hidden = dcu_static_or_runtime<kStaticHidden>(hidden_runtime);
    const int intermediate_hidden = dcu_static_or_runtime<kStaticIntermediate>(
        intermediate_hidden_runtime);
    const int num_experts = num_experts_per_rank_static * num_ranks_static;
    const int num_local_blocks = static_cast<int>(gridDim.x);
    auto* local_workspace = sym_buffers[rank_idx];
    auto* local_signals = signal_buffers[rank_idx];
    int* expert_counts = workspace_expert_counts(local_workspace);
    int* expert_task_pool = workspace_expert_task_pool(local_workspace, num_experts_per_rank_static);
    const int64_t max_tasks_per_expert =
        workspace_task_capacity_per_expert(num_ranks_static, num_max_tokens_per_rank);

    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
    if (blockIdx.x == 0)
        mega_moe_rank_barrier(signal_buffers, rank_idx, num_ranks_static);
    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

    for (int expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
         expert_idx < num_experts_per_rank_static;
         expert_idx += num_local_blocks * blockDim.x) {
        expert_counts[expert_idx] = 0;
    }
    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

    const int64_t total_route_tasks =
        static_cast<int64_t>(num_ranks_static) * num_tokens * num_topk_static;
    for (int64_t task = blockIdx.x * blockDim.x + threadIdx.x;
         task < total_route_tasks;
         task += static_cast<int64_t>(num_local_blocks) * blockDim.x) {
        const int topk_slot = static_cast<int>(task % num_topk_static);
        const int token_idx = static_cast<int>((task / num_topk_static) % num_tokens);
        const int source_rank =
            static_cast<int>(task / (static_cast<int64_t>(num_topk_static) * num_tokens));
        auto sections = get_sections(
            sym_buffers[source_rank], num_ranks_static, num_experts,
            num_max_tokens_per_rank, num_topk_static, hidden);

        const int64_t route_offset = static_cast<int64_t>(token_idx) * num_topk_static + topk_slot;
        const int64_t expert = sections.topk_idx[route_offset];
        const float route_weight = sections.topk_weights[route_offset];
        const bool valid = expert >= 0 && expert < num_experts && route_weight != 0.0f;
        const int owner_rank = valid ? static_cast<int>(expert / num_experts_per_rank_static) : -1;
        const bool owned_by_this_rank = valid && owner_rank == rank_idx;

        if (owned_by_this_rank) {
            const int local_expert =
                static_cast<int>(expert - static_cast<int64_t>(rank_idx) * num_experts_per_rank_static);
            const int pool_idx = atomicAdd(expert_counts + local_expert, 1);
            if (pool_idx < max_tasks_per_expert) {
                expert_task_pool[static_cast<int64_t>(local_expert) * max_tasks_per_expert + pool_idx] =
                    static_cast<int>(task);
            }
        } else if (!valid) {
            for (int h_idx = 0; h_idx < hidden; ++h_idx) {
                const int64_t partial_idx =
                    (static_cast<int64_t>(topk_slot) * num_max_tokens_per_rank + token_idx) * hidden + h_idx;
                sections.combine[partial_idx] = 0;
            }
        }
    }
    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

    if (cumulative_local_expert_recv_stats != nullptr) {
        for (int expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
             expert_idx < num_experts_per_rank_static;
             expert_idx += num_local_blocks * blockDim.x) {
            atomicAdd(cumulative_local_expert_recv_stats + expert_idx,
                      min(expert_counts[expert_idx], static_cast<int>(max_tasks_per_expert)));
        }
    }
    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

    {
        constexpr int route_tile_log2_m = kRouteTileLog2M;
        constexpr int route_tile_m = 1 << kRouteTileLog2M;
        constexpr int subtiles_per_tile = 1 << (kRouteTileLog2M - kDcuMmacTileMLog2);
        const auto layout = dcu_route_tile_scratch_layout(
            route_scratch_tiles, route_tile_m, hidden, intermediate_hidden);
        auto* all_x_fp8 = route_scratch + layout.x_fp8_offset;
        auto* all_act_bf16 = reinterpret_cast<uint16_t*>(route_scratch + layout.act_bf16_offset);
        auto* all_act_fp8 = route_scratch + layout.act_fp8_offset;
        auto* all_act_scale = reinterpret_cast<float*>(route_scratch + layout.act_scale_offset);
        auto* tile_x_row_ptrs =
            reinterpret_cast<const uint8_t**>(route_scratch + layout.tile_x_row_ptrs_offset);
        auto* tile_combine_row_ptrs =
            reinterpret_cast<uint16_t**>(route_scratch + layout.tile_combine_row_ptrs_offset);
        auto* tile_route_weights =
            reinterpret_cast<float*>(route_scratch + layout.tile_route_weight_offset);
        auto* tile_x_scales =
            reinterpret_cast<float*>(route_scratch + layout.tile_x_scale_offset);
        auto* tile_experts = reinterpret_cast<int*>(route_scratch + layout.tile_expert_offset);
        auto* tile_pool_bases = reinterpret_cast<int*>(route_scratch + layout.tile_pool_base_offset);
        auto* tile_counts = reinterpret_cast<int*>(route_scratch + layout.tile_count_offset);
        auto* l1_done_counts = reinterpret_cast<int*>(route_scratch + layout.l1_done_count_offset);
        auto* l2_queue = reinterpret_cast<int*>(route_scratch + layout.l2_queue_offset);
        auto* l2_queue_ready = reinterpret_cast<int*>(route_scratch + layout.l2_queue_ready_offset);
        auto* pipeline_counters = reinterpret_cast<int*>(route_scratch + layout.pipeline_counter_offset);
        auto* total_tiles_ptr = reinterpret_cast<int*>(route_scratch + layout.total_tiles_offset);

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            int total_tiles = 0;
            for (int expert_idx = 0; expert_idx < num_experts_per_rank_static; ++expert_idx) {
                const int expert_task_count =
                    min(expert_counts[expert_idx], static_cast<int>(max_tasks_per_expert));
                for (int pool_base = 0; pool_base < expert_task_count; pool_base += route_tile_m) {
                    if (total_tiles < route_scratch_tiles) {
                        tile_experts[total_tiles] = expert_idx;
                        tile_pool_bases[total_tiles] = pool_base;
                        tile_counts[total_tiles] = min(route_tile_m, expert_task_count - pool_base);
                        ++total_tiles;
                    }
                }
            }
            *total_tiles_ptr = total_tiles;
        }
        mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
        const int total_tiles = *total_tiles_ptr;
        const int total_subtiles = total_tiles * subtiles_per_tile;
        for (int subtile_id = blockIdx.x * blockDim.x + threadIdx.x;
             subtile_id < total_subtiles;
             subtile_id += num_local_blocks * blockDim.x) {
            l1_done_counts[subtile_id] = 0;
        }
        mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
        prepare_dcu_route_tile_metadata(
            sym_buffers, expert_task_pool, max_tasks_per_expert,
            total_tiles, route_tile_log2_m, tile_experts, tile_pool_bases, tile_counts,
            num_ranks_static, num_experts, num_max_tokens_per_rank,
            num_tokens, num_topk_static, hidden,
            num_local_blocks * blockDim.x,
            blockIdx.x * blockDim.x + threadIdx.x,
            tile_x_row_ptrs, tile_combine_row_ptrs,
            tile_route_weights, tile_x_scales);
        mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
        pull_dcu_route_tile_x_pool(
            tile_x_row_ptrs, tile_counts, total_tiles, route_tile_log2_m, hidden,
            num_local_blocks * blockDim.x,
            blockIdx.x * blockDim.x + threadIdx.x,
            all_x_fp8);
        mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

        const int wave_count = static_cast<int>(blockDim.x >> 6);
        const int inter_chunk = wave_count * 16 * kMTile16L1NChunksPerWave;
        const int num_inter_chunks = static_cast<int>(
            (static_cast<int64_t>(intermediate_hidden) + inter_chunk - 1) / inter_chunk);
        const int hidden_chunk = wave_count * 16 * kMTile16L2NChunksPerWave;
        const int num_hidden_chunks = static_cast<int>(
            (static_cast<int64_t>(hidden) + hidden_chunk - 1) / hidden_chunk);
        const int total_l2_tasks = total_subtiles * num_hidden_chunks;
        for (int task_id = blockIdx.x * blockDim.x + threadIdx.x;
             task_id < total_l2_tasks;
             task_id += num_local_blocks * blockDim.x) {
            l2_queue_ready[task_id] = 0;
        }
        if (blockIdx.x == 0 && threadIdx.x < 4)
            pipeline_counters[threadIdx.x] = 0;
        mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

        __shared__ int block_l1_task;
        __shared__ int block_quant_subtile;
        __shared__ int block_l2_queue_idx;
        __shared__ int block_stop;
        while (true) {
            if (threadIdx.x == 0)
                block_l1_task = atomicAdd(pipeline_counters + 0, 1);
            __syncthreads();

            if (block_l1_task < total_subtiles * num_inter_chunks) {
                const int subtile_task_id = block_l1_task / num_inter_chunks;
                const int chunk_id = block_l1_task - subtile_task_id * num_inter_chunks;
                const int tile_id = subtile_task_id / subtiles_per_tile;
                const int subtile_idx = subtile_task_id - tile_id * subtiles_per_tile;
                const int valid_rows = dcu_route_subtile_valid_rows(tile_counts[tile_id], subtile_idx);
                block_quant_subtile = -1;
                if (valid_rows <= 0) {
                    if (chunk_id == 0 && threadIdx.x == 0)
                        dcu_fetch_add_release_int(pipeline_counters + 3, num_hidden_chunks);
                } else {
                    const int64_t tile_meta_base = dcu_route_meta_base(tile_id, route_tile_log2_m, subtile_idx);
                    const int64_t act_offset =
                        ((static_cast<int64_t>(tile_id) << route_tile_log2_m) +
                         (static_cast<int64_t>(subtile_idx) << kDcuMmacTileMLog2)) * intermediate_hidden;
                    compute_route_mmac_mtile16_l1_chunk(
                        tile_experts[tile_id], chunk_id * inter_chunk,
                        valid_rows,
                        tile_meta_base,
                        hidden, intermediate_hidden, activation_clamp,
                        all_x_fp8, tile_route_weights, tile_x_scales,
                        l1_weights, l1_weights_sf,
                        all_act_bf16 + act_offset);
                    __threadfence();
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        const int done = dcu_fetch_add_release_int(
                            l1_done_counts + subtile_task_id, 1) + 1;
                        block_quant_subtile = (done == num_inter_chunks) ? subtile_task_id : -1;
                    }
                    __syncthreads();

                    if (block_quant_subtile >= 0) {
                        const int quant_tile_id = block_quant_subtile / subtiles_per_tile;
                        const int quant_subtile_idx = block_quant_subtile - quant_tile_id * subtiles_per_tile;
                        const int quant_valid_rows =
                            dcu_route_subtile_valid_rows(tile_counts[quant_tile_id], quant_subtile_idx);
                        const int64_t quant_act_offset =
                            ((static_cast<int64_t>(quant_tile_id) << route_tile_log2_m) +
                             (static_cast<int64_t>(quant_subtile_idx) << kDcuMmacTileMLog2)) *
                            intermediate_hidden;
                        const int64_t quant_scale_offset =
                            (static_cast<int64_t>(quant_tile_id) << route_tile_log2_m) +
                            (static_cast<int64_t>(quant_subtile_idx) << kDcuMmacTileMLog2);
                        quant_bf16_act_channelwise_mtile16_global(
                            all_act_bf16 + quant_act_offset,
                            all_act_fp8 + quant_act_offset,
                            all_act_scale + quant_scale_offset,
                            quant_valid_rows, intermediate_hidden);
                        __threadfence();
                        __syncthreads();
                        if (threadIdx.x == 0) {
                            const int queue_base = atomicAdd(pipeline_counters + 1, num_hidden_chunks);
                            for (int chunk_id = 0; chunk_id < num_hidden_chunks; ++chunk_id) {
                                const int queue_idx = queue_base + chunk_id;
                                l2_queue[queue_idx] = block_quant_subtile * num_hidden_chunks + chunk_id;
                                dcu_store_release_int(l2_queue_ready + queue_idx, 1);
                            }
                        }
                    }
                }
            }

            if (threadIdx.x == 0) {
                block_l2_queue_idx = -1;
                while (true) {
                    const int head = dcu_load_acquire_int(
                        reinterpret_cast<volatile int*>(pipeline_counters + 2));
                    const int tail = dcu_load_acquire_int(
                        reinterpret_cast<volatile int*>(pipeline_counters + 1));
                    if (head >= tail)
                        break;
                    if (atomicCAS(pipeline_counters + 2, head, head + 1) == head) {
                        block_l2_queue_idx = head;
                        break;
                    }
                }
            }
            __syncthreads();

            if (block_l2_queue_idx >= 0) {
                while (dcu_load_acquire_int(
                           reinterpret_cast<volatile int*>(l2_queue_ready + block_l2_queue_idx)) == 0) {}
                const int task_id = l2_queue[block_l2_queue_idx];
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

            if (threadIdx.x == 0) {
                block_stop = dcu_load_acquire_int(
                    reinterpret_cast<volatile int*>(pipeline_counters + 3)) >= total_l2_tasks;
            }
            __syncthreads();
            if (block_stop)
                break;
        }
        mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
    }

    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
    if (blockIdx.x == 0)
        mega_moe_rank_barrier(signal_buffers, rank_idx, num_ranks_static);
    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

    auto local_sections = get_sections(
        sym_buffers[rank_idx], num_ranks_static, num_experts,
        num_max_tokens_per_rank, num_topk_static, hidden);
    if ((hidden & 7) == 0) {
        constexpr int kBf16PerVec = 8;
        const int vecs_per_token = hidden / kBf16PerVec;
        const int64_t total_reduce_vecs = static_cast<int64_t>(num_tokens) * vecs_per_token;
        auto* y_vec = reinterpret_cast<uint4*>(y);
        for (int64_t task = blockIdx.x * blockDim.x + threadIdx.x;
             task < total_reduce_vecs;
             task += static_cast<int64_t>(num_local_blocks) * blockDim.x) {
            const int token_idx = static_cast<int>(task / vecs_per_token);
            const int vec_idx = static_cast<int>(task - static_cast<int64_t>(token_idx) * vecs_per_token);
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;
            float sum4 = 0.0f;
            float sum5 = 0.0f;
            float sum6 = 0.0f;
            float sum7 = 0.0f;
            for (int topk_slot = 0; topk_slot < num_topk_static; ++topk_slot) {
                const int64_t partial_row =
                    static_cast<int64_t>(topk_slot) * num_max_tokens_per_rank + token_idx;
                const auto packed =
                    reinterpret_cast<const uint4*>(local_sections.combine + partial_row * hidden)[vec_idx];
                sum0 += bf16_bits_to_float(static_cast<uint16_t>(packed.x));
                sum1 += bf16_bits_to_float(static_cast<uint16_t>(packed.x >> 16));
                sum2 += bf16_bits_to_float(static_cast<uint16_t>(packed.y));
                sum3 += bf16_bits_to_float(static_cast<uint16_t>(packed.y >> 16));
                sum4 += bf16_bits_to_float(static_cast<uint16_t>(packed.z));
                sum5 += bf16_bits_to_float(static_cast<uint16_t>(packed.z >> 16));
                sum6 += bf16_bits_to_float(static_cast<uint16_t>(packed.w));
                sum7 += bf16_bits_to_float(static_cast<uint16_t>(packed.w >> 16));
            }
            uint4 out;
            out.x = static_cast<uint32_t>(float_to_bf16_bits(sum0)) |
                    (static_cast<uint32_t>(float_to_bf16_bits(sum1)) << 16);
            out.y = static_cast<uint32_t>(float_to_bf16_bits(sum2)) |
                    (static_cast<uint32_t>(float_to_bf16_bits(sum3)) << 16);
            out.z = static_cast<uint32_t>(float_to_bf16_bits(sum4)) |
                    (static_cast<uint32_t>(float_to_bf16_bits(sum5)) << 16);
            out.w = static_cast<uint32_t>(float_to_bf16_bits(sum6)) |
                    (static_cast<uint32_t>(float_to_bf16_bits(sum7)) << 16);
            y_vec[task] = out;
        }
    } else {
        const int64_t total_reduce_tasks = static_cast<int64_t>(num_tokens) * hidden;
        for (int64_t task = blockIdx.x * blockDim.x + threadIdx.x;
             task < total_reduce_tasks;
             task += static_cast<int64_t>(num_local_blocks) * blockDim.x) {
            const int token_idx = static_cast<int>(task / hidden);
            const int h_idx = static_cast<int>(task % hidden);
            float sum = 0.0f;
            for (int topk_slot = 0; topk_slot < num_topk_static; ++topk_slot) {
                const int64_t partial_idx =
                    (static_cast<int64_t>(topk_slot) * num_max_tokens_per_rank + token_idx) * hidden + h_idx;
                sum += bf16_bits_to_float(local_sections.combine[partial_idx]);
            }
            y[static_cast<int64_t>(token_idx) * hidden + h_idx] = float_to_bf16_bits(sum);
        }
    }

    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
    if (blockIdx.x == 0)
        mega_moe_rank_barrier(signal_buffers, rank_idx, num_ranks_static);
    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
    return;
}

} // namespace deep_gemm::mega
