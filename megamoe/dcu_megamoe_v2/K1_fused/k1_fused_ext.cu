#include <hip/hip_runtime.h>

#include <cstdint>

#define DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN
#define DCU_MEGAMOE_V2_KERNEL_ONLY
#include "../../../csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp"

namespace {

template <int kBlockM, int kCUs, bool kMaskTinyStore>
void launch_k1_ll_symm_stage_impl(
    hip_bfloat16* out,
    const uint8_t* staged_x,
    const uint8_t* weight_pack5,
    const float* staged_x_scale,
    const float* weight_scale,
    const int32_t* problem_size,
    uint8_t* sym_buffer,
    int32_t* route_scratch_i32,
    int32_t* grid_barrier,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    int rows_aligned_per_expert,
    float* route_weights,
    int32_t* row_expert_out,
    int32_t* output_index,
    int64_t* row_combine_ptrs,
    uint8_t* local_topk_mask,
    int32_t* tail_tokens,
    int32_t* cumulative_local_expert_recv_stats,
    hipStream_t stream) {
    V2_K1_LowLatencyMaskedGroupGemmKernel<
        32, 4096, 4096, kBlockM, 256, 64, 4, kCUs, kMaskTinyStore, true, false>
        <<<dim3(kCUs), dim3(256), 0, stream>>>(
            out,
            staged_x,
            weight_pack5,
            staged_x_scale,
            weight_scale,
            problem_size,
            rows_aligned_per_expert,
            sym_buffer,
            route_scratch_i32,
            grid_barrier,
            rank_idx,
            num_ranks,
            num_global_experts,
            num_max_tokens_per_rank,
            num_topk,
            runtime_num_tokens,
            nullptr,
            0,
            route_weights,
            row_expert_out,
            output_index,
            row_combine_ptrs,
            local_topk_mask,
            tail_tokens,
            cumulative_local_expert_recv_stats);
}

}  // namespace

void dcu_megamoe_v2_launch_k1_ll_symm_stage_raw(
    hip_bfloat16* out,
    const uint8_t* staged_x,
    const uint8_t* weight_pack5,
    const float* staged_x_scale,
    const float* weight_scale,
    const int32_t* problem_size,
    uint8_t* sym_buffer,
    int32_t* route_scratch_i32,
    int32_t* grid_barrier,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    int rows_aligned_per_expert,
    int valid_rows_per_expert,
    int ll_block_m,
    int ll_cus,
    float* route_weights,
    int32_t* row_expert_out,
    int32_t* output_index,
    int64_t* row_combine_ptrs,
    uint8_t* local_topk_mask,
    int32_t* tail_tokens,
    int32_t* cumulative_local_expert_recv_stats,
    hipStream_t stream) {
    const bool mask_tiny_store = valid_rows_per_expert <= 16;
    if (ll_block_m == 32 && ll_cus == 64) {
        if (mask_tiny_store) {
            launch_k1_ll_symm_stage_impl<32, 64, true>(
                out, staged_x, weight_pack5, staged_x_scale, weight_scale,
                problem_size, sym_buffer, route_scratch_i32, grid_barrier,
                rank_idx, num_ranks, num_global_experts,
                num_max_tokens_per_rank, num_topk, runtime_num_tokens,
                rows_aligned_per_expert,
                route_weights, row_expert_out, output_index, row_combine_ptrs,
                local_topk_mask, tail_tokens,
                cumulative_local_expert_recv_stats,
                stream);
        } else {
            launch_k1_ll_symm_stage_impl<32, 64, false>(
                out, staged_x, weight_pack5, staged_x_scale, weight_scale,
                problem_size, sym_buffer, route_scratch_i32, grid_barrier,
                rank_idx, num_ranks, num_global_experts,
                num_max_tokens_per_rank, num_topk, runtime_num_tokens,
                rows_aligned_per_expert,
                route_weights, row_expert_out, output_index, row_combine_ptrs,
                local_topk_mask, tail_tokens,
                cumulative_local_expert_recv_stats,
                stream);
        }
    } else if (ll_block_m == 48 && ll_cus == 64) {
        launch_k1_ll_symm_stage_impl<48, 64, false>(
            out, staged_x, weight_pack5, staged_x_scale, weight_scale,
            problem_size, sym_buffer, route_scratch_i32, grid_barrier, rank_idx,
            num_ranks, num_global_experts, num_max_tokens_per_rank, num_topk,
            runtime_num_tokens, rows_aligned_per_expert, route_weights, row_expert_out,
            output_index, row_combine_ptrs, local_topk_mask, tail_tokens,
            cumulative_local_expert_recv_stats,
            stream);
    } else if (ll_block_m == 64 && ll_cus == 64) {
        launch_k1_ll_symm_stage_impl<64, 64, false>(
            out, staged_x, weight_pack5, staged_x_scale, weight_scale,
            problem_size, sym_buffer, route_scratch_i32, grid_barrier, rank_idx,
            num_ranks, num_global_experts, num_max_tokens_per_rank, num_topk,
            runtime_num_tokens, rows_aligned_per_expert, route_weights, row_expert_out,
            output_index, row_combine_ptrs, local_topk_mask, tail_tokens,
            cumulative_local_expert_recv_stats,
            stream);
    }
}

void dcu_megamoe_v2_launch_k1_normal_symm_stage_raw(
    hip_bfloat16* out,
    const uint8_t* staged_x,
    const uint8_t* weight_pack5,
    const float* staged_x_scale,
    const float* weight_scale,
    const int32_t* row_expert,
    uint8_t* sym_buffer,
    int32_t* route_scratch_i32,
    int32_t* grid_barrier,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    int rows_aligned_per_expert,
    int valid_rows_per_expert,
    int epoch,
    float* route_weights,
    int32_t* row_expert_out,
    int32_t* output_index,
    int64_t* row_combine_ptrs,
    uint8_t* local_topk_mask,
    int32_t* tail_tokens,
    int32_t* cumulative_local_expert_recv_stats,
    hipStream_t stream) {
    const int local_experts = num_global_experts / num_ranks;
    const int m = local_experts * rows_aligned_per_expert;
    constexpr int c_grid_x = 16;
    constexpr int c_stage_n_group = 4;
    constexpr int c_launch_grid_x =
        (c_grid_x + c_stage_n_group - 1) / c_stage_n_group;
    const int c_grid_y = (m + 255) / 256;

    V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
        256, 256, true, true, c_stage_n_group, 4096, 4096, false, false, false>
        <<<dim3(c_launch_grid_x, c_grid_y), dim3(768), 0, stream>>>(
            out,
            staged_x,
            weight_pack5,
            staged_x_scale,
            weight_scale,
            row_expert,
            m,
            4096,
            4096,
            rows_aligned_per_expert,
            valid_rows_per_expert,
            sym_buffer,
            grid_barrier,
            epoch,
            rank_idx,
            num_ranks,
            num_global_experts,
            num_max_tokens_per_rank,
            num_topk,
            runtime_num_tokens,
            nullptr,
            0,
            0,
            nullptr,
            nullptr,
            nullptr,
            0,
            route_scratch_i32,
            route_weights,
            row_expert_out,
            output_index,
            row_combine_ptrs,
            local_topk_mask,
            tail_tokens,
            cumulative_local_expert_recv_stats);
}
