#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdint>

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
    hipStream_t stream);

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
    hipStream_t stream);

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA/HIP");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t local_experts_checked(const int64_t num_ranks,
                              const int64_t num_global_experts) {
    TORCH_CHECK(num_ranks > 0 && num_global_experts % num_ranks == 0,
                "invalid V2 rank/expert shape");
    const int64_t local_experts = num_global_experts / num_ranks;
    TORCH_CHECK(local_experts == 32,
                "V2 C pack5 K1 is currently specialized for 32 local experts");
    return local_experts;
}

void check_common_k1_shape(
    const torch::Tensor& out,
    const torch::Tensor& staged_x,
    const torch::Tensor& weight_pack5,
    const torch::Tensor& staged_x_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& sym_buffer,
    const int64_t rows_aligned_per_expert,
    const int64_t num_ranks,
    const int64_t num_global_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_topk) {
    check_cuda_contiguous(out, "out");
    check_cuda_contiguous(staged_x, "staged_x");
    check_cuda_contiguous(weight_pack5, "weight_pack5");
    check_cuda_contiguous(staged_x_scale, "staged_x_scale");
    check_cuda_contiguous(weight_scale, "weight_scale");
    check_cuda_contiguous(sym_buffer, "sym_buffer");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be BF16");
    TORCH_CHECK(staged_x.scalar_type() == torch::kFloat8_e4m3fn,
                "staged_x must be FP8 e4m3fn");
    TORCH_CHECK(weight_pack5.scalar_type() == torch::kFloat8_e4m3fn,
                "weight_pack5 must be FP8 e4m3fn");
    TORCH_CHECK(staged_x_scale.scalar_type() == torch::kFloat32,
                "staged_x_scale must be FP32");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32,
                "weight_scale must be FP32");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be int8");
    const int64_t local_experts =
        local_experts_checked(num_ranks, num_global_experts);
    TORCH_CHECK(rows_aligned_per_expert > 0,
                "rows_aligned_per_expert must be positive");
    TORCH_CHECK(num_max_tokens_per_rank >= 0 && num_topk > 0,
                "invalid token/topk shape");
    TORCH_CHECK(out.dim() == 2 &&
                    out.size(0) >= local_experts * rows_aligned_per_expert &&
                    out.size(1) == 4096,
                "out must cover [local_experts * rows_aligned_per_expert, 4096]");
    TORCH_CHECK(staged_x.numel() >=
                    local_experts * rows_aligned_per_expert * 4096,
                "staged_x workspace is too small for K1");
    TORCH_CHECK(staged_x_scale.numel() >=
                    local_experts * rows_aligned_per_expert,
                "staged_x_scale workspace is too small for K1");
    TORCH_CHECK(weight_pack5.numel() >= local_experts * 4096LL * 4096LL,
                "weight_pack5 is too small for K1 L1 pack5");
    TORCH_CHECK(weight_scale.numel() >= local_experts * 4096LL,
                "weight_scale is too small for K1 L1");
}

void launch_k1_ll_symm_stage(
    const torch::Tensor& out,
    const torch::Tensor& staged_x,
    const torch::Tensor& weight_pack5,
    const torch::Tensor& staged_x_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& problem_size,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& route_scratch_i32,
    const torch::Tensor& grid_barrier,
    const torch::Tensor& route_weights,
    const torch::Tensor& row_expert_out,
    const torch::Tensor& output_index,
    const torch::Tensor& row_combine_ptrs,
    const torch::Tensor& local_topk_mask,
    const torch::Tensor& tail_tokens,
    const torch::Tensor& cumulative_local_expert_recv_stats,
    const int64_t rank_idx,
    const int64_t num_ranks,
    const int64_t num_global_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_topk,
    const int64_t num_tokens,
    const int64_t rows_aligned_per_expert,
    const int64_t valid_rows_per_expert,
    const int64_t ll_block_m,
    const int64_t ll_cus) {
    check_common_k1_shape(
        out, staged_x, weight_pack5, staged_x_scale, weight_scale, sym_buffer,
        rows_aligned_per_expert, num_ranks, num_global_experts,
        num_max_tokens_per_rank, num_topk);
    check_cuda_contiguous(problem_size, "problem_size");
    check_cuda_contiguous(route_scratch_i32, "route_scratch_i32");
    check_cuda_contiguous(grid_barrier, "grid_barrier");
    check_cuda_contiguous(route_weights, "route_weights");
    check_cuda_contiguous(row_expert_out, "row_expert_out");
    check_cuda_contiguous(output_index, "output_index");
    check_cuda_contiguous(row_combine_ptrs, "row_combine_ptrs");
    check_cuda_contiguous(local_topk_mask, "local_topk_mask");
    check_cuda_contiguous(tail_tokens, "tail_tokens");
    check_cuda_contiguous(cumulative_local_expert_recv_stats,
                          "cumulative_local_expert_recv_stats");
    TORCH_CHECK(problem_size.scalar_type() == torch::kInt &&
                    problem_size.numel() >= 32,
                "problem_size must be int32[>=32]");
    TORCH_CHECK(route_scratch_i32.scalar_type() == torch::kInt,
                "route_scratch_i32 must be int32");
    TORCH_CHECK(grid_barrier.scalar_type() == torch::kInt &&
                    grid_barrier.numel() >= 2,
                "grid_barrier must be int32[>=2]");
    const int64_t local_experts =
        local_experts_checked(num_ranks, num_global_experts);
    const int64_t rows = local_experts * rows_aligned_per_expert;
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32 &&
                    route_weights.numel() >= rows,
                "route_weights must be fp32 and cover K1 launch rows");
    TORCH_CHECK(row_expert_out.scalar_type() == torch::kInt &&
                    row_expert_out.numel() >= rows,
                "row_expert_out must be int32 and cover K1 launch rows");
    TORCH_CHECK(output_index.scalar_type() == torch::kInt &&
                    output_index.numel() >=
                        num_ranks * num_max_tokens_per_rank * num_topk,
                "output_index must be int32[num_ranks * max_tokens * topk]");
    TORCH_CHECK(row_combine_ptrs.scalar_type() == torch::kInt64 &&
                    row_combine_ptrs.numel() >= rows,
                "row_combine_ptrs must be int64 and cover K1 launch rows");
    TORCH_CHECK(local_topk_mask.scalar_type() == torch::kUInt8 &&
                    local_topk_mask.numel() >= num_max_tokens_per_rank,
                "local_topk_mask must be uint8[>=num_max_tokens_per_rank]");
    TORCH_CHECK(tail_tokens.scalar_type() == torch::kInt &&
                    tail_tokens.numel() >= num_max_tokens_per_rank,
                "tail_tokens must be int32[>=num_max_tokens_per_rank]");
    TORCH_CHECK(
        cumulative_local_expert_recv_stats.scalar_type() == torch::kInt &&
            (cumulative_local_expert_recv_stats.numel() == 0 ||
             cumulative_local_expert_recv_stats.numel() >= local_experts),
        "cumulative_local_expert_recv_stats must be empty or int32[local_experts]");
    TORCH_CHECK(rank_idx >= 0 && rank_idx < num_ranks, "invalid rank_idx");
    TORCH_CHECK(num_tokens >= -1 && num_tokens <= num_max_tokens_per_rank,
                "num_tokens must be -1 or in [0, num_max_tokens_per_rank]");
    TORCH_CHECK((ll_block_m == 32 || ll_block_m == 48 || ll_block_m == 64) &&
                    ll_cus == 64,
                "V2 K1 ll launcher currently supports ll_block_m in {32,48,64} with ll_cus=64");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dcu_megamoe_v2_launch_k1_ll_symm_stage_raw(
        reinterpret_cast<hip_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const uint8_t*>(staged_x.data_ptr()),
        reinterpret_cast<const uint8_t*>(weight_pack5.data_ptr()),
        staged_x_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        problem_size.data_ptr<int32_t>(),
        reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
        route_scratch_i32.data_ptr<int32_t>(),
        grid_barrier.data_ptr<int32_t>(),
        static_cast<int>(rank_idx),
        static_cast<int>(num_ranks),
        static_cast<int>(num_global_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk),
        static_cast<int>(num_tokens),
        static_cast<int>(rows_aligned_per_expert),
        static_cast<int>(valid_rows_per_expert),
        static_cast<int>(ll_block_m),
        static_cast<int>(ll_cus),
        route_weights.data_ptr<float>(),
        row_expert_out.data_ptr<int32_t>(),
        output_index.data_ptr<int32_t>(),
        row_combine_ptrs.data_ptr<int64_t>(),
        local_topk_mask.data_ptr<uint8_t>(),
        tail_tokens.data_ptr<int32_t>(),
        cumulative_local_expert_recv_stats.numel() == 0
            ? nullptr
            : cumulative_local_expert_recv_stats.data_ptr<int32_t>(),
        stream);
    TORCH_CHECK(hipGetLastError() == hipSuccess,
                "launch_k1_ll_symm_stage failed");
}

void launch_k1_normal_symm_stage(
    const torch::Tensor& out,
    const torch::Tensor& staged_x,
    const torch::Tensor& weight_pack5,
    const torch::Tensor& staged_x_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& row_expert,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& route_scratch_i32,
    const torch::Tensor& grid_barrier,
    const torch::Tensor& route_weights,
    const torch::Tensor& output_index,
    const torch::Tensor& row_combine_ptrs,
    const torch::Tensor& local_topk_mask,
    const torch::Tensor& tail_tokens,
    const torch::Tensor& cumulative_local_expert_recv_stats,
    const int64_t rank_idx,
    const int64_t num_ranks,
    const int64_t num_global_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_topk,
    const int64_t num_tokens,
    const int64_t rows_aligned_per_expert,
    const int64_t valid_rows_per_expert,
    const int64_t epoch) {
    check_common_k1_shape(
        out, staged_x, weight_pack5, staged_x_scale, weight_scale, sym_buffer,
        rows_aligned_per_expert, num_ranks, num_global_experts,
        num_max_tokens_per_rank, num_topk);
    check_cuda_contiguous(row_expert, "row_expert");
    check_cuda_contiguous(route_scratch_i32, "route_scratch_i32");
    check_cuda_contiguous(grid_barrier, "grid_barrier");
    check_cuda_contiguous(route_weights, "route_weights");
    check_cuda_contiguous(output_index, "output_index");
    check_cuda_contiguous(row_combine_ptrs, "row_combine_ptrs");
    check_cuda_contiguous(local_topk_mask, "local_topk_mask");
    check_cuda_contiguous(tail_tokens, "tail_tokens");
    check_cuda_contiguous(cumulative_local_expert_recv_stats,
                          "cumulative_local_expert_recv_stats");
    TORCH_CHECK(grid_barrier.scalar_type() == torch::kInt,
                "grid_barrier must be int32");
    TORCH_CHECK(valid_rows_per_expert > 0 &&
                    valid_rows_per_expert <= rows_aligned_per_expert,
                "invalid rows_per_expert contract");
    const int64_t local_experts =
        local_experts_checked(num_ranks, num_global_experts);
    const int64_t m = local_experts * rows_aligned_per_expert;
    TORCH_CHECK(row_expert.scalar_type() == torch::kInt &&
                    row_expert.numel() >= m,
                "row_expert must be int32 and cover launch rows");
    TORCH_CHECK(route_scratch_i32.scalar_type() == torch::kInt,
                "route_scratch_i32 must be int32");
    TORCH_CHECK(route_scratch_i32.numel() >=
                    32 + 2 * 32 * rows_aligned_per_expert,
                "route_scratch_i32 is too small for K1 normal metadata");
    TORCH_CHECK(num_tokens >= -1 && num_tokens <= num_max_tokens_per_rank,
                "num_tokens must be -1 or in [0, num_max_tokens_per_rank]");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32 &&
                    route_weights.numel() >= m,
                "route_weights must be fp32 and cover K1 launch rows");
    TORCH_CHECK(output_index.scalar_type() == torch::kInt &&
                    output_index.numel() >=
                        num_ranks * num_max_tokens_per_rank * num_topk,
                "output_index must be int32[num_ranks * max_tokens * topk]");
    TORCH_CHECK(row_combine_ptrs.scalar_type() == torch::kInt64 &&
                    row_combine_ptrs.numel() >= m,
                "row_combine_ptrs must be int64 and cover K1 launch rows");
    TORCH_CHECK(local_topk_mask.scalar_type() == torch::kUInt8 &&
                    local_topk_mask.numel() >= num_max_tokens_per_rank,
                "local_topk_mask must be uint8[>=num_max_tokens_per_rank]");
    TORCH_CHECK(tail_tokens.scalar_type() == torch::kInt &&
                    tail_tokens.numel() >= num_max_tokens_per_rank,
                "tail_tokens must be int32[>=num_max_tokens_per_rank]");
    TORCH_CHECK(
        cumulative_local_expert_recv_stats.scalar_type() == torch::kInt &&
            (cumulative_local_expert_recv_stats.numel() == 0 ||
             cumulative_local_expert_recv_stats.numel() >= local_experts),
        "cumulative_local_expert_recv_stats must be empty or int32[local_experts]");
    const int c_grid_y = static_cast<int>((m + 255) / 256);
    TORCH_CHECK(grid_barrier.numel() >= 2LL + 2LL * c_grid_y,
                "grid_barrier is too small for K1 normal row-stage");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dcu_megamoe_v2_launch_k1_normal_symm_stage_raw(
        reinterpret_cast<hip_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const uint8_t*>(staged_x.data_ptr()),
        reinterpret_cast<const uint8_t*>(weight_pack5.data_ptr()),
        staged_x_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        row_expert.data_ptr<int32_t>(),
        reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
        route_scratch_i32.data_ptr<int32_t>(),
        grid_barrier.data_ptr<int32_t>(),
        static_cast<int>(rank_idx),
        static_cast<int>(num_ranks),
        static_cast<int>(num_global_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk),
        static_cast<int>(num_tokens),
        static_cast<int>(rows_aligned_per_expert),
        static_cast<int>(valid_rows_per_expert),
        static_cast<int>(epoch),
        route_weights.data_ptr<float>(),
        row_expert.data_ptr<int32_t>(),
        output_index.data_ptr<int32_t>(),
        row_combine_ptrs.data_ptr<int64_t>(),
        local_topk_mask.data_ptr<uint8_t>(),
        tail_tokens.data_ptr<int32_t>(),
        cumulative_local_expert_recv_stats.numel() == 0
            ? nullptr
            : cumulative_local_expert_recv_stats.data_ptr<int32_t>(),
        stream);
    TORCH_CHECK(hipGetLastError() == hipSuccess,
                "launch_k1_normal_symm_stage failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_k1_ll_symm_stage", &launch_k1_ll_symm_stage,
          "Launch V2 low-latency C pack5 K1 dispatch-pull plus L1 groupgemm");
    m.def("launch_k1_normal_symm_stage", &launch_k1_normal_symm_stage,
          "Launch V2 normal C pack5 K1 dispatch-pull plus L1 groupgemm");
}
