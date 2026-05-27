#pragma once

#include <deep_gemm/common/mega_moe_dcu.cuh>
#include <deep_gemm/comm/mega_moe_dcu.cuh>
#include <deep_gemm/impls/mega_moe_dcu_task.cuh>
#include <deep_gemm/impls/mega_moe_dcu_tiles.cuh>
#include <deep_gemm/scheduler/mega_moe_dcu.cuh>

namespace deep_gemm::mega {

template <
    int kNumRanks_,
    int kTopK_,
    int kHidden_,
    int kIntermediate_,
    int kExpertsPerRank_>
struct DcuMegaMoeShape {
    static constexpr int kNumRanks = kNumRanks_;
    static constexpr int kTopK = kTopK_;
    static constexpr int kHidden = kHidden_;
    static constexpr int kIntermediate = kIntermediate_;
    static constexpr int kExpertsPerRank = kExpertsPerRank_;
    static constexpr int kNumExperts = kNumRanks * kExpertsPerRank;
};

enum class DcuMegaMoeL1Schedule {
    TileMajor,
    ChunkMajor,
    ExpertChunkMajor,
    ExpertWindowChunkMajor,
};

enum class DcuMegaMoeL2Schedule {
    SubtileMajor,
    ExpertHiddenMajor,
};

struct DcuMegaMoeL1TileMajorSchedule {
    static constexpr DcuMegaMoeL1Schedule kKind = DcuMegaMoeL1Schedule::TileMajor;
    static constexpr int kWindowMtiles = 0;
};

struct DcuMegaMoeL1ChunkMajorSchedule {
    static constexpr DcuMegaMoeL1Schedule kKind = DcuMegaMoeL1Schedule::ChunkMajor;
    static constexpr int kWindowMtiles = 0;
};

struct DcuMegaMoeL1ExpertChunkMajorSchedule {
    static constexpr DcuMegaMoeL1Schedule kKind = DcuMegaMoeL1Schedule::ExpertChunkMajor;
    static constexpr int kWindowMtiles = 0;
};

template <int kWindowMtiles_>
struct DcuMegaMoeL1ExpertWindowChunkMajorSchedule {
    static_assert(kWindowMtiles_ > 0);
    static constexpr DcuMegaMoeL1Schedule kKind =
        DcuMegaMoeL1Schedule::ExpertWindowChunkMajor;
    static constexpr int kWindowMtiles = kWindowMtiles_;
};

struct DcuMegaMoeL2SubtileMajorSchedule {
    static constexpr DcuMegaMoeL2Schedule kKind = DcuMegaMoeL2Schedule::SubtileMajor;
    static constexpr int kGroupSubtiles = 0;
};

template <int kGroupSubtiles_ = 0>
struct DcuMegaMoeL2ExpertHiddenMajorSchedule {
    static_assert(kGroupSubtiles_ >= 0);
    static constexpr DcuMegaMoeL2Schedule kKind =
        DcuMegaMoeL2Schedule::ExpertHiddenMajor;
    static constexpr int kGroupSubtiles = kGroupSubtiles_;
};

template <
    int kRouteTileLog2M_,
    typename L1TilePolicy_,
    typename L2TilePolicy_,
    typename L1SchedulePolicy_,
    typename L2SchedulePolicy_,
    int kThreads_ = 256,
    int kBlocks_ = 128>
struct DcuMegaMoeTilePolicy {
    using L1TilePolicy = L1TilePolicy_;
    using L2TilePolicy = L2TilePolicy_;
    using L1SchedulePolicy = L1SchedulePolicy_;
    using L2SchedulePolicy = L2SchedulePolicy_;
    static constexpr int kRouteTileLog2M = kRouteTileLog2M_;
    static constexpr DcuMegaMoeL1Schedule kL1Schedule = L1SchedulePolicy::kKind;
    static constexpr int kL1ScheduleWindowMtiles = L1SchedulePolicy::kWindowMtiles;
    static constexpr DcuMegaMoeL2Schedule kL2Schedule = L2SchedulePolicy::kKind;
    static constexpr int kL2ScheduleGroupSubtiles = L2SchedulePolicy::kGroupSubtiles;
    static constexpr int kThreads = kThreads_;
    static constexpr int kBlocks = kBlocks_;
};

template <typename Shape_, typename TilePolicy_>
struct DcuMegaMoeKernelConfig {
    using Shape = Shape_;
    using TilePolicy = TilePolicy_;
    using L1TilePolicy = typename TilePolicy::L1TilePolicy;
    using L2TilePolicy = typename TilePolicy::L2TilePolicy;
    using L1TileConfig = DcuResolvedMmacTileConfig<L1TilePolicy, TilePolicy::kThreads>;
    using L2TileConfig = DcuResolvedMmacTileConfig<L2TilePolicy, TilePolicy::kThreads>;
    using L1SchedulePolicy = typename TilePolicy::L1SchedulePolicy;
    using L2SchedulePolicy = typename TilePolicy::L2SchedulePolicy;
    static constexpr int kRouteTileLog2M = TilePolicy::kRouteTileLog2M;
    static constexpr DcuMegaMoeL1Schedule kL1Schedule = TilePolicy::kL1Schedule;
    static constexpr int kL1ScheduleWindowMtiles = TilePolicy::kL1ScheduleWindowMtiles;
    static constexpr DcuMegaMoeL2Schedule kL2Schedule = TilePolicy::kL2Schedule;
    static constexpr int kL2ScheduleGroupSubtiles = TilePolicy::kL2ScheduleGroupSubtiles;
    static constexpr int kThreads = TilePolicy::kThreads;
    static constexpr int kBlocks = TilePolicy::kBlocks;
};

using DcuMegaMoeEp8Shape = DcuMegaMoeShape<8, 6, 4096, 2048, 32>;

template <
    typename L1TilePolicy_,
    typename L2TilePolicy_,
    int kRouteTileLog2M_ = kDcuRouteTileMDefaultLog2,
    typename L1SchedulePolicy_ = DcuMegaMoeL1ExpertChunkMajorSchedule,
    typename L2SchedulePolicy_ = DcuMegaMoeL2ExpertHiddenMajorSchedule<0>,
    int kThreads_ = 256,
    int kBlocks_ = 192>
using DcuMegaMoeDefaultTilePolicy = DcuMegaMoeTilePolicy<
    kRouteTileLog2M_, L1TilePolicy_, L2TilePolicy_,
    L1SchedulePolicy_, L2SchedulePolicy_, kThreads_, kBlocks_>;

using DcuMegaMoeDefaultPolicy =
    DcuMegaMoeDefaultTilePolicy<
        DcuL1N256Pad64TilePolicy, DcuL2N512TilePolicy,
        kDcuRouteTileMDefaultLog2,
        DcuMegaMoeL1ExpertChunkMajorSchedule,
        DcuMegaMoeL2ExpertHiddenMajorSchedule<16>,
        256, 144>;
using DcuMegaMoeEp8Config =
    DcuMegaMoeKernelConfig<DcuMegaMoeEp8Shape, DcuMegaMoeDefaultPolicy>;

template <DcuMegaMoeL1Schedule kSchedule, int kNumExperts, int kWindowMtiles>
__device__ static inline void dcu_decode_l1_task(
    const int task_id,
    const int total_l1_mtiles,
    const int num_inter_chunks,
    const int* expert_l1_task_offsets,
    int* mtile_task_id,
    int* chunk_id) {
    if constexpr (kSchedule == DcuMegaMoeL1Schedule::ChunkMajor) {
        *chunk_id = task_id / total_l1_mtiles;
        *mtile_task_id = task_id - *chunk_id * total_l1_mtiles;
    } else if constexpr (
        kSchedule == DcuMegaMoeL1Schedule::ExpertChunkMajor ||
        kSchedule == DcuMegaMoeL1Schedule::ExpertWindowChunkMajor) {
        int expert_idx = 0;
#pragma unroll
        for (int e = 0; e < kNumExperts; ++e) {
            if (task_id < expert_l1_task_offsets[e + 1]) {
                expert_idx = e;
                break;
            }
        }
        const int expert_task_begin = expert_l1_task_offsets[expert_idx];
        const int expert_task_count =
            expert_l1_task_offsets[expert_idx + 1] - expert_task_begin;
        const int expert_mtiles = expert_task_count / num_inter_chunks;
        const int local_task = task_id - expert_task_begin;
        if constexpr (kSchedule == DcuMegaMoeL1Schedule::ExpertWindowChunkMajor) {
            constexpr int window_mtiles_const = kWindowMtiles > 0 ? kWindowMtiles : 1;
            const int window_task_capacity = window_mtiles_const * num_inter_chunks;
            const int window_idx = local_task / window_task_capacity;
            const int window_mtile_start = window_idx * window_mtiles_const;
            const int window_mtiles =
                min(window_mtiles_const, expert_mtiles - window_mtile_start);
            const int window_task = local_task - window_idx * window_task_capacity;
            *chunk_id = window_task / window_mtiles;
            const int local_mtile_task =
                window_mtile_start + window_task - *chunk_id * window_mtiles;
            *mtile_task_id = expert_task_begin / num_inter_chunks + local_mtile_task;
        } else {
            *chunk_id = local_task / expert_mtiles;
            const int local_mtile_task = local_task - *chunk_id * expert_mtiles;
            *mtile_task_id = expert_task_begin / num_inter_chunks + local_mtile_task;
        }
    } else {
        *mtile_task_id = task_id / num_inter_chunks;
        *chunk_id = task_id - *mtile_task_id * num_inter_chunks;
    }
}

template <typename KernelConfig>
__device__ static inline void dcu_enqueue_l2_ready_for_subtile(
    const int subtile_task_id,
    const int quant_tile_id,
    const int num_inter_chunks,
    const int num_hidden_chunks,
    const int* tile_experts,
    const int* expert_l1_task_offsets,
    int* expert_quant_done_counts,
    int* l2_group_done_counts,
    int* pipeline_counters,
    int* l2_queue,
    int* l2_queue_ready) {
    if constexpr (KernelConfig::kL2Schedule == DcuMegaMoeL2Schedule::ExpertHiddenMajor) {
        const int quant_expert = tile_experts[quant_tile_id];
        const int expert_task_begin = expert_l1_task_offsets[quant_expert];
        const int expert_task_count =
            expert_l1_task_offsets[quant_expert + 1] - expert_task_begin;
        const int expert_subtiles = expert_task_count / num_inter_chunks;
        const int subtile_begin = expert_task_begin / num_inter_chunks;
        int group_subtile_begin = 0;
        int group_subtiles = expert_subtiles;
        bool enqueue_group = false;
        if constexpr (KernelConfig::kL2ScheduleGroupSubtiles > 0) {
            constexpr int group_limit = KernelConfig::kL2ScheduleGroupSubtiles;
            const int local_subtile = subtile_task_id - subtile_begin;
            const int group_idx = local_subtile / group_limit;
            group_subtile_begin = group_idx * group_limit;
            group_subtiles = min(group_limit, expert_subtiles - group_subtile_begin);
            const int done = dcu_fetch_add_release_int(
                l2_group_done_counts + subtile_begin + group_idx, 1) + 1;
            enqueue_group = (done == group_subtiles);
        } else {
            const int done = dcu_fetch_add_release_int(
                expert_quant_done_counts + quant_expert, 1) + 1;
            enqueue_group = (done == expert_subtiles);
        }
        if (enqueue_group) {
            const int queue_base = atomicAdd(
                pipeline_counters + kDcuPipelineL2QueueTail,
                group_subtiles * num_hidden_chunks);
            for (int hidden_chunk_id = 0;
                 hidden_chunk_id < num_hidden_chunks;
                 ++hidden_chunk_id) {
                for (int subtile_offset = 0;
                     subtile_offset < group_subtiles;
                     ++subtile_offset) {
                    const int queue_idx =
                        queue_base + hidden_chunk_id * group_subtiles + subtile_offset;
                    const int queued_subtile_task_id =
                        subtile_begin + group_subtile_begin + subtile_offset;
                    l2_queue[queue_idx] =
                        queued_subtile_task_id * num_hidden_chunks + hidden_chunk_id;
                    dcu_store_release_int(l2_queue_ready + queue_idx, 1);
                }
            }
        }
    } else {
        const int queue_base = atomicAdd(
            pipeline_counters + kDcuPipelineL2QueueTail, num_hidden_chunks);
        for (int hidden_chunk_id = 0;
             hidden_chunk_id < num_hidden_chunks;
             ++hidden_chunk_id) {
            const int queue_idx = queue_base + hidden_chunk_id;
            l2_queue[queue_idx] = subtile_task_id * num_hidden_chunks + hidden_chunk_id;
            dcu_store_release_int(l2_queue_ready + queue_idx, 1);
        }
    }
}

template <typename KernelConfig, bool kUseRuntimeNumTokens>
__global__ __launch_bounds__(512) void mega_moe_multirank_persistent_w8a8_channelwise_kernel(
    uint16_t* y,
    uint8_t** sym_buffers,
    int** signal_buffers,
    uint8_t* route_scratch,
    const int64_t route_scratch_tiles,
    const uint8_t* l1_weights, const float* l1_weights_sf,
    const uint8_t* l2_weights, const float* l2_weights_sf,
    int* cumulative_local_expert_recv_stats,
    const int rank_idx,
    const int num_max_tokens_per_rank,
    const int launch_num_tokens,
    const int* runtime_num_tokens,
    const float activation_clamp) {
    using Shape = typename KernelConfig::Shape;
    using L1TileConfig = typename KernelConfig::L1TileConfig;
    using L2TileConfig = typename KernelConfig::L2TileConfig;
    static_assert(Shape::kNumRanks > 0 && Shape::kTopK > 0 && Shape::kHidden > 0 &&
                  Shape::kIntermediate > 0 && Shape::kNumExperts > 0 &&
                  Shape::kExpertsPerRank > 0,
                  "DCU MegaMoE fused kernel is shape-specialized.");
    constexpr int num_ranks_static = Shape::kNumRanks;
    constexpr int num_experts_per_rank_static = Shape::kExpertsPerRank;
    constexpr int num_topk_static = Shape::kTopK;
    constexpr int hidden = Shape::kHidden;
    constexpr int intermediate_hidden = Shape::kIntermediate;
    constexpr int num_experts = Shape::kNumExperts;
    auto local_sections = get_sections(
        sym_buffers[rank_idx], num_ranks_static, num_experts,
        num_max_tokens_per_rank, num_topk_static, hidden);
    int num_tokens = launch_num_tokens;
    if constexpr (kUseRuntimeNumTokens) {
        num_tokens = min(max(*runtime_num_tokens, 0), launch_num_tokens);
    }
    const int num_local_blocks = static_cast<int>(gridDim.x);
    auto* local_signals = signal_buffers[rank_idx];
    int* expert_counts = route_scratch_expert_counts(route_scratch);
    int* expert_task_pool = route_scratch_expert_task_pool(
        route_scratch, num_experts_per_rank_static);
    const int64_t max_tasks_per_expert =
        workspace_task_capacity_per_expert(num_ranks_static, num_max_tokens_per_rank);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *local_sections.num_tokens = num_tokens;
    }

    for (int expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
         expert_idx < num_experts_per_rank_static;
         expert_idx += num_local_blocks * blockDim.x) {
        expert_counts[expert_idx] = 0;
    }

    if (blockIdx.x == 0)
        mega_moe_rank_barrier(signal_buffers, rank_idx, num_ranks_static);
    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

    __shared__ int block_uniform_num_tokens;
    if (threadIdx.x == 0) {
        int uniform = 1;
        for (int source_rank = 0; source_rank < num_ranks_static; ++source_rank) {
            auto sections = get_sections(
                sym_buffers[source_rank], num_ranks_static, num_experts,
                num_max_tokens_per_rank, num_topk_static, hidden);
            int source_num_tokens = *sections.num_tokens;
            source_num_tokens = min(max(source_num_tokens, 0), num_max_tokens_per_rank);
            if (source_num_tokens != num_tokens) {
                uniform = 0;
            }
        }
        block_uniform_num_tokens = uniform;
    }
    __syncthreads();
    const bool uniform_num_tokens = block_uniform_num_tokens != 0;
    const int task_num_tokens_stride =
        uniform_num_tokens ? num_tokens : num_max_tokens_per_rank;
    if (uniform_num_tokens) {
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
    } else {
        const int64_t total_route_tasks =
            static_cast<int64_t>(num_ranks_static) * num_max_tokens_per_rank * num_topk_static;
        for (int64_t task = blockIdx.x * blockDim.x + threadIdx.x;
             task < total_route_tasks;
             task += static_cast<int64_t>(num_local_blocks) * blockDim.x) {
            const int topk_slot = static_cast<int>(task % num_topk_static);
            const int token_idx =
                static_cast<int>((task / num_topk_static) % num_max_tokens_per_rank);
            const int source_rank = static_cast<int>(
                task / (static_cast<int64_t>(num_topk_static) * num_max_tokens_per_rank));
            auto sections = get_sections(
                sym_buffers[source_rank], num_ranks_static, num_experts,
                num_max_tokens_per_rank, num_topk_static, hidden);
            int source_num_tokens = *sections.num_tokens;
            source_num_tokens = min(max(source_num_tokens, 0), num_max_tokens_per_rank);
            if (token_idx >= source_num_tokens) {
                continue;
            }

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

    {
        constexpr int route_tile_log2_m = KernelConfig::kRouteTileLog2M;
        constexpr int route_tile_m = 1 << KernelConfig::kRouteTileLog2M;
        constexpr int subtiles_per_tile =
            1 << (KernelConfig::kRouteTileLog2M - kDcuMmacTileMLog2);
        uint8_t* route_tile_scratch = route_tile_scratch_base(
            route_scratch, num_ranks_static, num_experts, num_max_tokens_per_rank);
        const auto layout = dcu_route_tile_scratch_layout(
            route_scratch_tiles, route_tile_m, hidden, intermediate_hidden);
        auto* all_x_fp8 = route_tile_scratch + layout.x_fp8_offset;
        auto* all_act_bf16 = reinterpret_cast<uint16_t*>(route_tile_scratch + layout.act_bf16_offset);
        auto* all_act_fp8 = route_tile_scratch + layout.act_fp8_offset;
        auto* all_act_scale = reinterpret_cast<float*>(route_tile_scratch + layout.act_scale_offset);
        auto* all_act_chunk_amax =
            reinterpret_cast<float*>(route_tile_scratch + layout.act_chunk_amax_offset);
        auto* tile_x_row_ptrs =
            reinterpret_cast<const uint8_t**>(route_tile_scratch + layout.tile_x_row_ptrs_offset);
        auto* tile_combine_row_ptrs =
            reinterpret_cast<uint16_t**>(route_tile_scratch + layout.tile_combine_row_ptrs_offset);
        auto* tile_route_weights =
            reinterpret_cast<float*>(route_tile_scratch + layout.tile_route_weight_offset);
        auto* tile_x_scales =
            reinterpret_cast<float*>(route_tile_scratch + layout.tile_x_scale_offset);
        auto* tile_experts = reinterpret_cast<int*>(route_tile_scratch + layout.tile_expert_offset);
        auto* tile_pool_bases = reinterpret_cast<int*>(route_tile_scratch + layout.tile_pool_base_offset);
        auto* tile_counts = reinterpret_cast<int*>(route_tile_scratch + layout.tile_count_offset);
        auto* expert_l1_task_offsets =
            reinterpret_cast<int*>(route_tile_scratch + layout.expert_l1_task_offset);
        auto* expert_quant_done_counts =
            reinterpret_cast<int*>(route_tile_scratch + layout.expert_quant_done_count_offset);
        auto* l2_group_done_counts =
            reinterpret_cast<int*>(route_tile_scratch + layout.l2_group_done_count_offset);
        auto* tile_pull_done = reinterpret_cast<int*>(route_tile_scratch + layout.tile_pull_done_offset);
        auto* l1_done_counts = reinterpret_cast<int*>(route_tile_scratch + layout.l1_done_count_offset);
        auto* l2_queue = reinterpret_cast<int*>(route_tile_scratch + layout.l2_queue_offset);
        auto* l2_queue_ready = reinterpret_cast<int*>(route_tile_scratch + layout.l2_queue_ready_offset);
        auto* pipeline_counters = reinterpret_cast<int*>(route_tile_scratch + layout.pipeline_counter_offset);
        auto* total_tiles_ptr = reinterpret_cast<int*>(route_tile_scratch + layout.total_tiles_offset);

        constexpr bool use_l1_chunk_local_quant =
            L1TileConfig::kTileM == kDcuMmacTileM &&
            L2TileConfig::kKStageBytes == L1TileConfig::kTileN;
        constexpr int l1_mtiles_per_route_tile = subtiles_per_tile;
        const int inter_chunk = L1TileConfig::kTileN;
        const int num_inter_chunks = static_cast<int>(
            (static_cast<int64_t>(intermediate_hidden) + inter_chunk - 1) / inter_chunk);
        const int hidden_chunk = L2TileConfig::kTileN;
        const int num_hidden_chunks = static_cast<int>(
            (static_cast<int64_t>(hidden) + hidden_chunk - 1) / hidden_chunk);

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            int total_tiles = 0;
            int total_l1_tasks_by_expert = 0;
            for (int expert_idx = 0; expert_idx < num_experts_per_rank_static; ++expert_idx) {
                expert_l1_task_offsets[expert_idx] = total_l1_tasks_by_expert;
                const int expert_task_count =
                    min(expert_counts[expert_idx], static_cast<int>(max_tasks_per_expert));
                const int expert_tile_start = total_tiles;
                for (int pool_base = 0; pool_base < expert_task_count; pool_base += route_tile_m) {
                    if (total_tiles < route_scratch_tiles) {
                        tile_experts[total_tiles] = expert_idx;
                        tile_pool_bases[total_tiles] = pool_base;
                        tile_counts[total_tiles] = min(route_tile_m, expert_task_count - pool_base);
                        ++total_tiles;
                    }
                }
                total_l1_tasks_by_expert +=
                    (total_tiles - expert_tile_start) * l1_mtiles_per_route_tile * num_inter_chunks;
            }
            expert_l1_task_offsets[num_experts_per_rank_static] = total_l1_tasks_by_expert;
            *total_tiles_ptr = total_tiles;
        }
        mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);
        const int total_tiles = *total_tiles_ptr;
        const int total_subtiles = total_tiles * subtiles_per_tile;
        const int total_l1_mtiles = total_tiles * l1_mtiles_per_route_tile;
        const int total_l1_tasks = total_l1_mtiles * num_inter_chunks;
        const int total_l2_tasks = total_subtiles * num_hidden_chunks;
        for (int expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
             expert_idx < num_experts_per_rank_static;
             expert_idx += num_local_blocks * blockDim.x) {
            expert_quant_done_counts[expert_idx] = 0;
        }
        for (int subtile_id = blockIdx.x * blockDim.x + threadIdx.x;
             subtile_id < total_subtiles;
             subtile_id += num_local_blocks * blockDim.x) {
            l1_done_counts[subtile_id] = 0;
            l2_group_done_counts[subtile_id] = 0;
        }
        for (int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
             tile_id < total_tiles;
             tile_id += num_local_blocks * blockDim.x) {
            tile_pull_done[tile_id] = 0;
        }
        for (int task_id = blockIdx.x * blockDim.x + threadIdx.x;
             task_id < total_l2_tasks;
             task_id += num_local_blocks * blockDim.x) {
            l2_queue_ready[task_id] = 0;
        }
        if (blockIdx.x == 0 && threadIdx.x < kDcuPipelineCounterCount)
            pipeline_counters[threadIdx.x] = 0;
        prepare_dcu_route_tile_metadata(
            sym_buffers, expert_task_pool, max_tasks_per_expert,
            total_tiles, route_tile_log2_m, tile_experts, tile_pool_bases, tile_counts,
            num_ranks_static, num_experts, num_max_tokens_per_rank,
            task_num_tokens_stride, num_topk_static, hidden,
            num_local_blocks * blockDim.x,
            blockIdx.x * blockDim.x + threadIdx.x,
            tile_x_row_ptrs, tile_combine_row_ptrs,
            tile_route_weights, tile_x_scales);
        mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

        __shared__ int block_pull_tile;
        __shared__ int block_l1_task;
        __shared__ int block_quant_subtile;
        __shared__ int block_l2_queue_idx;
        __shared__ int block_stop;
        constexpr int l1_local_quant_vecs = use_l1_chunk_local_quant
            ? L1TileConfig::kL1ChunkLocalQuantVecs
            : 0;
        constexpr int l1_stage_or_quant_vecs =
            L1TileConfig::kLdsStageTotalVecs > l1_local_quant_vecs
                ? L1TileConfig::kLdsStageTotalVecs
                : l1_local_quant_vecs;
        constexpr int lds_stage_vecs =
            l1_stage_or_quant_vecs > L2TileConfig::kLdsStageTotalVecs
                ? l1_stage_or_quant_vecs
                : L2TileConfig::kLdsStageTotalVecs;
        __shared__ uint4 block_lds_a_stage[lds_stage_vecs];
        while (true) {
            if (threadIdx.x == 0) {
                const int pull_head = dcu_load_acquire_int(
                    reinterpret_cast<volatile int*>(
                        pipeline_counters + kDcuPipelinePullTileHead));
                block_pull_tile = pull_head < total_tiles
                    ? atomicAdd(pipeline_counters + kDcuPipelinePullTileHead, 1)
                    : -1;
            }
            __syncthreads();
            if (block_pull_tile >= 0 && block_pull_tile < total_tiles) {
                pull_one_dcu_route_tile_x_pool(
                    tile_x_row_ptrs, tile_counts, block_pull_tile,
                    route_tile_log2_m, hidden,
                    blockDim.x, threadIdx.x,
                    all_x_fp8);
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0)
                    dcu_store_release_int(tile_pull_done + block_pull_tile, 1);
            }

            if (threadIdx.x == 0) {
                const int l1_head = dcu_load_acquire_int(
                    reinterpret_cast<volatile int*>(
                        pipeline_counters + kDcuPipelineL1TaskHead));
                block_l1_task = l1_head < total_l1_tasks
                    ? atomicAdd(pipeline_counters + kDcuPipelineL1TaskHead, 1)
                    : -1;
            }
            __syncthreads();

            if (block_l1_task >= 0 && block_l1_task < total_l1_tasks) {
                int subtile_task_id;
                int chunk_id;
                dcu_decode_l1_task<
                    KernelConfig::kL1Schedule, num_experts_per_rank_static,
                    KernelConfig::kL1ScheduleWindowMtiles>(
                    block_l1_task, total_l1_mtiles, num_inter_chunks,
                    expert_l1_task_offsets, &subtile_task_id, &chunk_id);
                const int tile_id = subtile_task_id / subtiles_per_tile;
                const int subtile_idx = subtile_task_id - tile_id * subtiles_per_tile;
                const int valid_rows = dcu_route_subtile_valid_rows(tile_counts[tile_id], subtile_idx);
                while (dcu_load_acquire_int(
                           reinterpret_cast<volatile int*>(tile_pull_done + tile_id)) == 0) {}
                block_quant_subtile = -1;
                if (valid_rows <= 0) {
                    if (chunk_id == 0 && threadIdx.x == 0) {
                        dcu_enqueue_l2_ready_for_subtile<KernelConfig>(
                            subtile_task_id, tile_id, num_inter_chunks, num_hidden_chunks,
                            tile_experts, expert_l1_task_offsets,
                            expert_quant_done_counts, l2_group_done_counts,
                            pipeline_counters, l2_queue, l2_queue_ready);
                    }
                } else {
                    const int64_t tile_meta_base =
                        dcu_route_meta_base(tile_id, route_tile_log2_m, subtile_idx);
                    const int64_t act_offset =
                        ((static_cast<int64_t>(tile_id) << route_tile_log2_m) +
                         (static_cast<int64_t>(subtile_idx) << kDcuMmacTileMLog2)) *
                        intermediate_hidden;
                    compute_route_mmac_mtile16_l1_chunk<
                        L1TileConfig, use_l1_chunk_local_quant>(
                        tile_experts[tile_id], chunk_id * inter_chunk,
                        valid_rows,
                        tile_meta_base,
                        hidden, intermediate_hidden, activation_clamp,
                        all_x_fp8, tile_route_weights, tile_x_scales,
                        l1_weights, l1_weights_sf,
                        all_act_bf16 + act_offset,
                        all_act_fp8 + act_offset,
                        all_act_chunk_amax,
                        num_inter_chunks,
                        block_lds_a_stage);
                    __threadfence();
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        const int done = dcu_fetch_add_release_int(
                            l1_done_counts + subtile_task_id, 1) + 1;
                        block_quant_subtile =
                            (done == num_inter_chunks) ? subtile_task_id : -1;
                    }
                }
                __syncthreads();
                if (block_quant_subtile >= 0) {
                    const int quant_tile_id = block_quant_subtile / subtiles_per_tile;
                    const int quant_subtile_idx =
                        block_quant_subtile - quant_tile_id * subtiles_per_tile;
                    const int quant_valid_rows =
                        dcu_route_subtile_valid_rows(tile_counts[quant_tile_id], quant_subtile_idx);
                    const int64_t quant_act_offset =
                        ((static_cast<int64_t>(quant_tile_id) << route_tile_log2_m) +
                         (static_cast<int64_t>(quant_subtile_idx) << kDcuMmacTileMLog2)) *
                        intermediate_hidden;
                    const int64_t quant_scale_offset =
                        (static_cast<int64_t>(quant_tile_id) << route_tile_log2_m) +
                        (static_cast<int64_t>(quant_subtile_idx) << kDcuMmacTileMLog2);
                    if constexpr (!use_l1_chunk_local_quant) {
                        quant_bf16_act_channelwise_mtile16_global_with_chunk_amax(
                            all_act_bf16 + quant_act_offset,
                            all_act_fp8 + quant_act_offset,
                            all_act_scale + quant_scale_offset,
                            all_act_chunk_amax + quant_scale_offset * num_inter_chunks,
                            quant_valid_rows, intermediate_hidden, num_inter_chunks);
                        __threadfence();
                        __syncthreads();
                    } else {
                        __threadfence();
                        __syncthreads();
                    }
                    if (threadIdx.x == 0) {
                        dcu_enqueue_l2_ready_for_subtile<KernelConfig>(
                            block_quant_subtile, quant_tile_id,
                            num_inter_chunks, num_hidden_chunks,
                            tile_experts, expert_l1_task_offsets,
                            expert_quant_done_counts, l2_group_done_counts,
                            pipeline_counters, l2_queue, l2_queue_ready);
                    }
                }
            }

            dcu_run_l2_queue_task_if_ready<
                L2TileConfig, use_l1_chunk_local_quant, L1TileConfig::kTileN>(
                &block_l2_queue_idx,
                pipeline_counters,
                l2_queue, l2_queue_ready,
                num_hidden_chunks, subtiles_per_tile,
                tile_counts, tile_experts,
                route_tile_log2_m,
                hidden_chunk, hidden, intermediate_hidden,
                tile_combine_row_ptrs,
                l2_weights, l2_weights_sf,
                all_act_fp8, all_act_scale, all_act_chunk_amax, num_inter_chunks,
                block_lds_a_stage);

            if (threadIdx.x == 0) {
                block_stop = dcu_load_acquire_int(
                    reinterpret_cast<volatile int*>(
                        pipeline_counters + kDcuPipelineL2Done)) >= total_l2_tasks;
            }
            __syncthreads();
            if (block_stop)
                break;
        }
    }

    if (blockIdx.x == 0)
        mega_moe_rank_barrier(signal_buffers, rank_idx, num_ranks_static);
    mega_moe_local_blocks_barrier(local_signals, rank_idx, num_local_blocks);

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
            out.x = pack2_f32_to_bf16_bits(sum0, sum1);
            out.y = pack2_f32_to_bf16_bits(sum2, sum3);
            out.z = pack2_f32_to_bf16_bits(sum4, sum5);
            out.w = pack2_f32_to_bf16_bits(sum6, sum7);
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

    if (blockIdx.x == 0)
        mega_moe_rank_barrier(signal_buffers, rank_idx, num_ranks_static);
    return;
}

} // namespace deep_gemm::mega
