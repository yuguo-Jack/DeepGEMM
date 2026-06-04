#include <hip/hip_runtime.h>

#include <cstdint>

#define DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN
#define DCU_MEGAMOE_V2_KERNEL_ONLY
#include "../../../csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp"

namespace {

template <int kBlockM, int kCUs, bool kMaskTinyStore>
void launch_k3_ll_rowptr_tail_reduce_impl(
    hip_bfloat16* y,
    const uint8_t* act_fp8,
    const uint8_t* weight_pack5,
    const float* act_scale,
    const float* weight_scale,
    const int32_t* problem_size,
    uint8_t* sym_buffer,
    int32_t* grid_barrier,
    const int64_t* row_output_ptrs,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    int rows_aligned_per_expert,
    hipStream_t stream) {
    V2_K1_LowLatencyMaskedGroupGemmKernel<
        32, 4096, 2048, kBlockM, 256, 64, 4, kCUs, kMaskTinyStore, false, true>
        <<<dim3(kCUs), dim3(256), 0, stream>>>(
            y,
            act_fp8,
            weight_pack5,
            act_scale,
            weight_scale,
            problem_size,
            rows_aligned_per_expert,
            sym_buffer,
            nullptr,
            grid_barrier,
            rank_idx,
            num_ranks,
            num_global_experts,
            num_max_tokens_per_rank,
            num_topk,
            runtime_num_tokens,
            row_output_ptrs,
            1);
}

}  // namespace

void dcu_megamoe_v2_launch_k3_ll_rowptr_tail_reduce_raw(
    hip_bfloat16* y,
    const uint8_t* act_fp8,
    const uint8_t* weight_pack5,
    const float* act_scale,
    const float* weight_scale,
    const int32_t* problem_size,
    uint8_t* sym_buffer,
    int32_t* grid_barrier,
    const int64_t* row_output_ptrs,
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
    hipStream_t stream) {
    const bool mask_tiny_store = valid_rows_per_expert <= 16;
    if (ll_block_m == 32 && ll_cus == 64) {
        if (mask_tiny_store) {
            launch_k3_ll_rowptr_tail_reduce_impl<32, 64, true>(
                y, act_fp8, weight_pack5, act_scale, weight_scale, problem_size,
                sym_buffer, grid_barrier, row_output_ptrs, rank_idx, num_ranks,
                num_global_experts, num_max_tokens_per_rank, num_topk,
                runtime_num_tokens, rows_aligned_per_expert, stream);
        } else {
            launch_k3_ll_rowptr_tail_reduce_impl<32, 64, false>(
                y, act_fp8, weight_pack5, act_scale, weight_scale, problem_size,
                sym_buffer, grid_barrier, row_output_ptrs, rank_idx, num_ranks,
                num_global_experts, num_max_tokens_per_rank, num_topk,
                runtime_num_tokens, rows_aligned_per_expert, stream);
        }
    } else if (ll_block_m == 48 && ll_cus == 64) {
        launch_k3_ll_rowptr_tail_reduce_impl<48, 64, false>(
            y, act_fp8, weight_pack5, act_scale, weight_scale, problem_size,
            sym_buffer, grid_barrier, row_output_ptrs, rank_idx, num_ranks,
            num_global_experts, num_max_tokens_per_rank, num_topk,
            runtime_num_tokens, rows_aligned_per_expert, stream);
    } else if (ll_block_m == 64 && ll_cus == 64) {
        launch_k3_ll_rowptr_tail_reduce_impl<64, 64, false>(
            y, act_fp8, weight_pack5, act_scale, weight_scale, problem_size,
            sym_buffer, grid_barrier, row_output_ptrs, rank_idx, num_ranks,
            num_global_experts, num_max_tokens_per_rank, num_topk,
            runtime_num_tokens, rows_aligned_per_expert, stream);
    }
}

void dcu_megamoe_v2_launch_k3_normal_copy_stage_tail_reduce_raw(
    hip_bfloat16* l2_workspace,
    const uint8_t* act_fp8,
    const uint8_t* weight_pack5,
    const float* act_scale,
    const float* weight_scale,
    const int32_t* row_expert,
    uint8_t* sym_buffer,
    int32_t* grid_barrier,
    const int64_t* row_output_ptrs,
    const uint8_t* local_topk_mask,
    hip_bfloat16* tail_out,
    const int32_t* tail_tokens,
    int tail_token_count,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    int rows_aligned_per_expert,
    int valid_rows_per_expert,
    int k3_copy_workers,
    int epoch,
    hipStream_t stream) {
    const int local_experts = num_global_experts / num_ranks;
    const int m = local_experts * rows_aligned_per_expert;
    constexpr int c_grid_x = 16;
    const int c_grid_y = (m + 255) / 256;
    const int c_k3_copy_rows = (k3_copy_workers + c_grid_x - 1) / c_grid_x;

    V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
        256, 256, true, false, 1, 4096, 2048, false, true>
        <<<dim3(c_grid_x, c_grid_y + c_k3_copy_rows), dim3(768), 0, stream>>>(
            l2_workspace,
            act_fp8,
            weight_pack5,
            act_scale,
            weight_scale,
            row_expert,
            m,
            4096,
            2048,
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
            row_output_ptrs,
            k3_copy_workers,
            1,
            local_topk_mask,
            tail_out,
            tail_tokens,
            tail_token_count);
}
