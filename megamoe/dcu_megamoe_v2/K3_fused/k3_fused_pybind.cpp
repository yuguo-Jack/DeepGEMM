#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdint>

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
    hipStream_t stream);

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
                "V2 C pack5 K3 is currently specialized for 32 local experts");
    return local_experts;
}

int64_t check_common_k3_shape(
    const torch::Tensor& act_fp8,
    const torch::Tensor& weight_pack5,
    const torch::Tensor& act_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& row_output_ptrs,
    const int64_t rows_aligned_per_expert,
    const int64_t valid_rows_per_expert,
    const int64_t num_ranks,
    const int64_t num_global_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_topk) {
    check_cuda_contiguous(act_fp8, "act_fp8");
    check_cuda_contiguous(weight_pack5, "weight_pack5");
    check_cuda_contiguous(act_scale, "act_scale");
    check_cuda_contiguous(weight_scale, "weight_scale");
    check_cuda_contiguous(sym_buffer, "sym_buffer");
    check_cuda_contiguous(row_output_ptrs, "row_output_ptrs");
    TORCH_CHECK(act_fp8.scalar_type() == torch::kFloat8_e4m3fn,
                "act_fp8 must be FP8 e4m3fn");
    TORCH_CHECK(weight_pack5.scalar_type() == torch::kFloat8_e4m3fn,
                "weight_pack5 must be FP8 e4m3fn");
    TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32,
                "act_scale must be FP32");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32,
                "weight_scale must be FP32");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be int8");
    TORCH_CHECK(row_output_ptrs.scalar_type() == torch::kInt64,
                "row_output_ptrs must be int64");
    const int64_t local_experts =
        local_experts_checked(num_ranks, num_global_experts);
    const int64_t rows = local_experts * rows_aligned_per_expert;
    TORCH_CHECK(rows_aligned_per_expert > 0 &&
                    valid_rows_per_expert > 0 &&
                    valid_rows_per_expert <= rows_aligned_per_expert,
                "invalid rows_per_expert contract");
    TORCH_CHECK(num_max_tokens_per_rank >= 0 && num_topk > 0,
                "invalid token/topk shape");
    TORCH_CHECK(act_fp8.dim() == 2 && act_fp8.size(0) >= rows &&
                    act_fp8.size(1) == 2048,
                "act_fp8 must cover [rows, 2048]");
    TORCH_CHECK(act_scale.numel() >= rows,
                "act_scale must cover every K3 grouped row");
    TORCH_CHECK(row_output_ptrs.numel() >= rows,
                "row_output_ptrs must cover every K3 grouped row");
    TORCH_CHECK(weight_pack5.numel() >= local_experts * 4096LL * 2048LL,
                "weight_pack5 is too small for K3 L2 pack5");
    TORCH_CHECK(weight_scale.numel() >= local_experts * 4096LL,
                "weight_scale is too small for K3 L2");
    return rows;
}

void launch_k3_ll_rowptr_tail_reduce(
    const torch::Tensor& y,
    const torch::Tensor& act_fp8,
    const torch::Tensor& weight_pack5,
    const torch::Tensor& act_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& problem_size,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& grid_barrier,
    const torch::Tensor& row_output_ptrs,
    const int64_t rank_idx,
    const int64_t num_ranks,
    const int64_t num_global_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_topk,
    const int64_t runtime_num_tokens,
    const int64_t rows_aligned_per_expert,
    const int64_t valid_rows_per_expert,
    const int64_t ll_block_m,
    const int64_t ll_cus) {
    check_common_k3_shape(
        act_fp8, weight_pack5, act_scale, weight_scale, sym_buffer,
        row_output_ptrs, rows_aligned_per_expert, valid_rows_per_expert,
        num_ranks, num_global_experts, num_max_tokens_per_rank, num_topk);
    check_cuda_contiguous(y, "y");
    check_cuda_contiguous(problem_size, "problem_size");
    check_cuda_contiguous(grid_barrier, "grid_barrier");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "y must be BF16");
    TORCH_CHECK(y.dim() == 2 && y.size(0) >= num_max_tokens_per_rank &&
                    y.size(1) == 4096,
                "K3 ll tail-reduce writes num_max_tokens_per_rank rows of y");
    TORCH_CHECK(problem_size.scalar_type() == torch::kInt &&
                    problem_size.numel() >= 32,
                "problem_size must be int32[>=32]");
    TORCH_CHECK(grid_barrier.scalar_type() == torch::kInt &&
                    grid_barrier.numel() >= 2,
                "grid_barrier must be int32[>=2]");
    TORCH_CHECK((ll_block_m == 32 || ll_block_m == 48 || ll_block_m == 64) &&
                    ll_cus == 64,
                "V2 K3 ll launcher currently supports ll_block_m in {32,48,64} with ll_cus=64");
    TORCH_CHECK(runtime_num_tokens >= -1 &&
                    runtime_num_tokens <= num_max_tokens_per_rank,
                "runtime_num_tokens must be -1 or in [0, num_max_tokens_per_rank]");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dcu_megamoe_v2_launch_k3_ll_rowptr_tail_reduce_raw(
        reinterpret_cast<hip_bfloat16*>(y.data_ptr()),
        reinterpret_cast<const uint8_t*>(act_fp8.data_ptr()),
        reinterpret_cast<const uint8_t*>(weight_pack5.data_ptr()),
        act_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        problem_size.data_ptr<int32_t>(),
        reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
        grid_barrier.data_ptr<int32_t>(),
        row_output_ptrs.data_ptr<int64_t>(),
        static_cast<int>(rank_idx),
        static_cast<int>(num_ranks),
        static_cast<int>(num_global_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk),
        static_cast<int>(runtime_num_tokens),
        static_cast<int>(rows_aligned_per_expert),
        static_cast<int>(valid_rows_per_expert),
        static_cast<int>(ll_block_m),
        static_cast<int>(ll_cus),
        stream);
    TORCH_CHECK(hipGetLastError() == hipSuccess,
                "launch_k3_ll_rowptr_tail_reduce failed");
}

void launch_k3_normal_copy_stage_tail_reduce(
    const torch::Tensor& l2_workspace,
    const torch::Tensor& act_fp8,
    const torch::Tensor& weight_pack5,
    const torch::Tensor& act_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& row_expert,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& grid_barrier,
    const torch::Tensor& row_output_ptrs,
    const torch::Tensor& local_topk_mask,
    const torch::Tensor& tail_out,
    const torch::Tensor& tail_tokens,
    const int64_t rank_idx,
    const int64_t num_ranks,
    const int64_t num_global_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_topk,
    const int64_t runtime_num_tokens,
    const int64_t tail_token_count,
    const int64_t rows_aligned_per_expert,
    const int64_t valid_rows_per_expert,
    const int64_t k3_copy_workers,
    const int64_t epoch) {
    const int64_t rows = check_common_k3_shape(
        act_fp8, weight_pack5, act_scale, weight_scale, sym_buffer,
        row_output_ptrs, rows_aligned_per_expert, valid_rows_per_expert,
        num_ranks, num_global_experts, num_max_tokens_per_rank, num_topk);
    check_cuda_contiguous(l2_workspace, "l2_workspace");
    check_cuda_contiguous(row_expert, "row_expert");
    check_cuda_contiguous(grid_barrier, "grid_barrier");
    check_cuda_contiguous(local_topk_mask, "local_topk_mask");
    check_cuda_contiguous(tail_out, "tail_out");
    check_cuda_contiguous(tail_tokens, "tail_tokens");
    TORCH_CHECK(l2_workspace.scalar_type() == torch::kBFloat16,
                "l2_workspace must be BF16");
    TORCH_CHECK(l2_workspace.dim() == 2 && l2_workspace.size(0) >= rows &&
                    l2_workspace.size(1) == 4096,
                "l2_workspace must cover [rows, 4096]");
    TORCH_CHECK(row_expert.scalar_type() == torch::kInt &&
                    row_expert.numel() >= rows,
                "row_expert must be int32 and cover launch rows");
    TORCH_CHECK(grid_barrier.scalar_type() == torch::kInt,
                "grid_barrier must be int32");
    TORCH_CHECK(local_topk_mask.scalar_type() == torch::kUInt8 &&
                    local_topk_mask.numel() >=
                        num_max_tokens_per_rank * num_topk,
                "local_topk_mask must be uint8[num_max_tokens_per_rank * topk]");
    TORCH_CHECK(tail_out.scalar_type() == torch::kBFloat16 &&
                    tail_out.dim() == 2 &&
                    tail_out.size(0) >= num_max_tokens_per_rank &&
                    tail_out.size(1) == 4096,
                "tail_out must cover [num_max_tokens_per_rank, 4096]");
    TORCH_CHECK(tail_tokens.scalar_type() == torch::kInt,
                "tail_tokens must be int32");
    TORCH_CHECK(k3_copy_workers > 0 && k3_copy_workers <= 16,
                "k3_copy_workers must be in [1, 16]");
    TORCH_CHECK(runtime_num_tokens >= -1 &&
                    runtime_num_tokens <= num_max_tokens_per_rank,
                "runtime_num_tokens must be -1 or in [0, num_max_tokens_per_rank]");
    TORCH_CHECK(tail_token_count >= -1 &&
                    tail_token_count <= num_max_tokens_per_rank,
                "tail_token_count must be -1 or in [0, num_max_tokens_per_rank]");
    const int c_grid_y = static_cast<int>((rows + 255) / 256);
    TORCH_CHECK(grid_barrier.numel() >= 16LL * c_grid_y + 2,
                "grid_barrier is too small for K3 copy-stage");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dcu_megamoe_v2_launch_k3_normal_copy_stage_tail_reduce_raw(
        reinterpret_cast<hip_bfloat16*>(l2_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(act_fp8.data_ptr()),
        reinterpret_cast<const uint8_t*>(weight_pack5.data_ptr()),
        act_scale.data_ptr<float>(),
        weight_scale.data_ptr<float>(),
        row_expert.data_ptr<int32_t>(),
        reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
        grid_barrier.data_ptr<int32_t>(),
        row_output_ptrs.data_ptr<int64_t>(),
        local_topk_mask.data_ptr<uint8_t>(),
        reinterpret_cast<hip_bfloat16*>(tail_out.data_ptr()),
        tail_tokens.numel() == 0 ? nullptr : tail_tokens.data_ptr<int32_t>(),
        static_cast<int>(tail_token_count),
        static_cast<int>(rank_idx),
        static_cast<int>(num_ranks),
        static_cast<int>(num_global_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk),
        static_cast<int>(runtime_num_tokens),
        static_cast<int>(rows_aligned_per_expert),
        static_cast<int>(valid_rows_per_expert),
        static_cast<int>(k3_copy_workers),
        static_cast<int>(epoch),
        stream);
    TORCH_CHECK(hipGetLastError() == hipSuccess,
                "launch_k3_normal_copy_stage_tail_reduce failed");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_k3_ll_rowptr_tail_reduce",
          &launch_k3_ll_rowptr_tail_reduce,
          "Launch V2 low-latency C pack5 K3 L2 plus row-combine tail reduce");
    m.def("launch_k3_normal_copy_stage_tail_reduce",
          &launch_k3_normal_copy_stage_tail_reduce,
          "Launch V2 normal C pack5 K3 L2 plus copy-stage tail reduce");
}
