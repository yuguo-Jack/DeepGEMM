#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "k1_v3_pack5_groupgemm_impl.cuh"

void dcu_megamoe_v3_launch_k1_ll_symm_stage_pack5(
    hip_bfloat16* out,
    const uint8_t* staged_x,
    const uint8_t* weight_pack5,
    const float* staged_x_scale,
    const float* weight_scale,
    const int32_t* problem_size,
    uint8_t* sym_buffer,
    int32_t* route_scratch_i32,
    int32_t* grid_barrier,
    int barrier_epoch,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int l1_rows,
    int runtime_num_tokens,
    int rows_aligned_per_expert,
    int valid_rows_per_expert,
    int ll_block_m,
    int ll_cus,
    float* route_weights,
    int32_t* m_indices,
    int32_t* output_index,
    int64_t* row_combine_ptrs,
    uint8_t* local_topk_mask,
    int32_t* tail_tokens,
    int32_t* local_expert_stats,
    bool enable_start_rank_barrier,
    int32_t* tail_done_counter,
    const int32_t* graph_runtime_num_tokens_for_barrier,
    int32_t* graph_runtime_num_tokens_out,
    int32_t* graph_tail_signal_generation_out,
    int graph_max_tokens,
    hipStream_t stream) {
    (void)problem_size;
    const bool mask_tiny_store = valid_rows_per_expert <= 16;
#define DCU_MEGAMOE_V3_LAUNCH_K1_LL(                                            \
    EXPERTS, N, K, BLOCK_M, CUS, MASK_TINY_STORE, PARALLEL_STAGE_COPY)            \
    V3_K1_LowLatencyMaskedGroupGemmKernel<                                      \
        EXPERTS, N, K, BLOCK_M, ((K) == 7168 ? 128 : 256), 64, 4, CUS,           \
        MASK_TINY_STORE,                                                        \
        PARALLEL_STAGE_COPY>                                                     \
        <<<dim3(CUS), dim3(256), 0, stream>>>(                                  \
            out, staged_x, weight_pack5, staged_x_scale, weight_scale,           \
            rows_aligned_per_expert, sym_buffer, route_scratch_i32,              \
            grid_barrier, barrier_epoch, rank_idx,                               \
            num_ranks, num_global_experts, num_max_tokens_per_rank, num_topk,    \
            runtime_num_tokens, route_weights, m_indices, output_index,          \
            row_combine_ptrs, local_topk_mask, tail_tokens, local_expert_stats,  \
            enable_start_rank_barrier, tail_done_counter,                        \
            graph_runtime_num_tokens_for_barrier, graph_runtime_num_tokens_out,  \
            graph_tail_signal_generation_out, graph_max_tokens)

#define DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_SHAPE(EXPERTS, N, K)                    \
    do {                                                                        \
        if (ll_block_m == 32 && ll_cus == 64) {                                  \
            if (mask_tiny_store) {                                               \
                DCU_MEGAMOE_V3_LAUNCH_K1_LL(EXPERTS, N, K, 32, 64, true, true);  \
            } else {                                                             \
                DCU_MEGAMOE_V3_LAUNCH_K1_LL(EXPERTS, N, K, 32, 64, false, true); \
            }                                                                    \
        } else if (ll_block_m == 48 && ll_cus == 64) {                           \
            DCU_MEGAMOE_V3_LAUNCH_K1_LL(EXPERTS, N, K, 48, 64, false, false);    \
        } else if (ll_block_m == 64 && ll_cus == 64) {                           \
            if (mask_tiny_store) {                                               \
                DCU_MEGAMOE_V3_LAUNCH_K1_LL(EXPERTS, N, K, 64, 64, true, false); \
            } else {                                                             \
                DCU_MEGAMOE_V3_LAUNCH_K1_LL(EXPERTS, N, K, 64, 64, false, false);\
            }                                                                    \
        }                                                                        \
    } while (0)

#define DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(EXPERTS)                        \
    do {                                                                        \
        if (hidden == 4096 && l1_rows == 4096) {                                 \
            DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_SHAPE(EXPERTS, 4096, 4096);          \
        } else if (hidden == 7168 && l1_rows == 6144) {                          \
            DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_SHAPE(EXPERTS, 6144, 7168);          \
        }                                                                        \
    } while (0)

    const int local_experts =
        num_ranks > 0 ? (num_global_experts / num_ranks) : num_global_experts;
    if (local_experts == 8) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(8);
    } else if (local_experts == 12) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(12);
    } else if (local_experts == 16) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(16);
    } else if (local_experts == 24) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(24);
    } else if (local_experts == 32) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(32);
    } else if (local_experts == 48) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(48);
    }
#undef DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS
#undef DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_SHAPE
#undef DCU_MEGAMOE_V3_LAUNCH_K1_LL
}

void dcu_megamoe_v3_launch_k1_ll_stage_only_pack5(
    const uint8_t* staged_x,
    const float* staged_x_scale,
    uint8_t* sym_buffer,
    int32_t* route_scratch_i32,
    int32_t* grid_barrier,
    int barrier_epoch,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int l1_rows,
    int runtime_num_tokens,
    int rows_aligned_per_expert,
    int ll_block_m,
    int ll_cus,
    float* route_weights,
    int32_t* m_indices,
    int32_t* output_index,
    int64_t* row_combine_ptrs,
    uint8_t* local_topk_mask,
    int32_t* tail_tokens,
    int32_t* local_expert_stats,
    bool enable_start_rank_barrier,
    int32_t* tail_done_counter,
    const int32_t* graph_runtime_num_tokens_for_barrier,
    int32_t* graph_runtime_num_tokens_out,
    int32_t* graph_tail_signal_generation_out,
    int graph_max_tokens,
    hipStream_t stream) {
#define DCU_MEGAMOE_V3_LAUNCH_K1_LL_STAGE(EXPERTS, K, BLOCK_M, PARALLEL_STAGE_COPY) \
    V3_K1_LowLatencyStageOnlyKernel<EXPERTS, K, BLOCK_M, PARALLEL_STAGE_COPY>    \
        <<<dim3(ll_cus), dim3(256), 0, stream>>>(                                \
            staged_x, staged_x_scale, rows_aligned_per_expert, sym_buffer,       \
            route_scratch_i32, grid_barrier, barrier_epoch, rank_idx,            \
            num_ranks, num_global_experts, num_max_tokens_per_rank, num_topk,    \
            runtime_num_tokens, route_weights, m_indices, output_index,          \
            row_combine_ptrs, local_topk_mask, tail_tokens, local_expert_stats,  \
            enable_start_rank_barrier, tail_done_counter,                        \
            graph_runtime_num_tokens_for_barrier, graph_runtime_num_tokens_out,  \
            graph_tail_signal_generation_out, graph_max_tokens)

#define DCU_MEGAMOE_V3_LAUNCH_K1_LL_STAGE_FOR_PRO_EXPERTS(EXPERTS)              \
    do {                                                                        \
        if (hidden == 7168 && l1_rows == 6144 &&                                 \
            ll_block_m == 48 && ll_cus == 64) {                                  \
            DCU_MEGAMOE_V3_LAUNCH_K1_LL_STAGE(EXPERTS, 7168, 48, false);         \
        }                                                                        \
    } while (0)

    const int local_experts =
        num_ranks > 0 ? (num_global_experts / num_ranks) : num_global_experts;
    if (local_experts == 12) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_STAGE_FOR_PRO_EXPERTS(12);
    } else if (local_experts == 24) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_STAGE_FOR_PRO_EXPERTS(24);
    } else if (local_experts == 48) {
        DCU_MEGAMOE_V3_LAUNCH_K1_LL_STAGE_FOR_PRO_EXPERTS(48);
    }
#undef DCU_MEGAMOE_V3_LAUNCH_K1_LL_STAGE_FOR_PRO_EXPERTS
#undef DCU_MEGAMOE_V3_LAUNCH_K1_LL_STAGE
}
