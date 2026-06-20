#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdlib>
#include <cstdint>

#include "k3_v3_pack5_groupgemm_impl.cuh"

template <int kBlockM, bool kTailReduce>
static inline void launch_v3_k3_ll_rowptr(
    hip_bfloat16* output_workspace,
    const uint8_t* act_fp8,
    const float* act_scale,
    const int32_t* actual_m,
    const uint8_t* weight_pack5,
    const float* weight_scale,
    const int64_t* row_combine_ptrs,
    int rows_per_expert,
    uint8_t* sym_buffer,
    int32_t* done_counter,
    const int64_t* signal_addrs,
    hip_bfloat16* reduce_y,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_tokens,
    const int32_t* runtime_num_tokens,
    int num_topk,
    int done_target,
    int reduce_blocks,
    const int32_t* signal_generation_ptr,
    hipStream_t stream) {
    const dim3 block(256);
    constexpr int kCUs = 64;
    const dim3 grid(kTailReduce ? kCUs + reduce_blocks : kCUs);
    V3_K3_LowLatencyMaskedGroupGemmKernel<
        32, 4096, 2048, kBlockM, 256, 64, 4, kCUs, true, false, kTailReduce>
        <<<grid, block, 0, stream>>>(
            output_workspace,
            act_fp8,
            weight_pack5,
            act_scale,
            weight_scale,
            actual_m,
            rows_per_expert,
            row_combine_ptrs,
            sym_buffer,
            done_counter,
            signal_addrs,
            reinterpret_cast<uint16_t*>(reduce_y),
            num_ranks,
            num_experts,
            num_max_tokens_per_rank,
            num_tokens,
            runtime_num_tokens,
            num_topk,
            1,
            signal_generation_ptr,
            done_target,
            reduce_blocks);
}

void dcu_megamoe_v3_launch_k3_ll_combine_pack5(
    hip_bfloat16* output_workspace,
    const uint8_t* act_fp8,
    const float* act_scale,
    const int32_t* row_expert,
    const uint8_t* weight_pack5,
    const float* weight_scale,
    const int64_t* row_combine_ptrs,
    uint8_t* sym_buffer,
    int32_t* done_counter,
    int64_t* signal_addrs,
    hip_bfloat16* reduce_y,
    int total_rows,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_tokens,
    const int32_t* runtime_num_tokens,
    int num_topk,
    int hidden,
    int tail_reduce,
    int ll_block_m,
    const int32_t* signal_generation_ptr,
    hipStream_t stream) {
    if (hidden != 4096) {
        (void)output_workspace;
        (void)act_fp8;
        (void)act_scale;
        (void)row_expert;
        (void)weight_pack5;
        (void)weight_scale;
        (void)row_combine_ptrs;
        (void)signal_generation_ptr;
        (void)stream;
        return;
    }
    const int local_experts =
        num_ranks > 0 ? (num_experts / num_ranks) : num_experts;
    const int rows_per_expert =
        local_experts > 0 ? (total_rows / local_experts) : 0;
    if (rows_per_expert <= 0 || total_rows % local_experts != 0)
        return;
    const int reduce_blocks =
        tail_reduce ? (num_max_tokens_per_rank <= 2048 ? 64 : 128) : 0;
    const int done_target = tail_reduce ? 64 : 0;
#define DCU_MEGAMOE_V3_LAUNCH_LL(BLOCK_M, TAIL)                                \
    launch_v3_k3_ll_rowptr<(BLOCK_M), (TAIL)>(                                 \
        output_workspace, act_fp8, act_scale, row_expert, weight_pack5,         \
        weight_scale,                                                           \
        row_combine_ptrs, rows_per_expert, sym_buffer, done_counter,            \
        signal_addrs, reduce_y, num_ranks, num_experts,                        \
        num_max_tokens_per_rank, num_tokens, runtime_num_tokens, num_topk,      \
        done_target, reduce_blocks, signal_generation_ptr, stream)
    if (ll_block_m == 64) {
        if (tail_reduce) {
            DCU_MEGAMOE_V3_LAUNCH_LL(64, true);
        } else {
            DCU_MEGAMOE_V3_LAUNCH_LL(64, false);
        }
    } else if (ll_block_m == 48) {
        if (tail_reduce) {
            DCU_MEGAMOE_V3_LAUNCH_LL(48, true);
        } else {
            DCU_MEGAMOE_V3_LAUNCH_LL(48, false);
        }
    } else {
        if (tail_reduce) {
            DCU_MEGAMOE_V3_LAUNCH_LL(32, true);
        } else {
            DCU_MEGAMOE_V3_LAUNCH_LL(32, false);
        }
    }
#undef DCU_MEGAMOE_V3_LAUNCH_LL
    (void)sym_buffer;
    (void)done_counter;
    (void)signal_addrs;
    (void)reduce_y;
    (void)num_ranks;
    (void)num_experts;
    (void)num_max_tokens_per_rank;
    (void)num_tokens;
    (void)num_topk;
    (void)ll_block_m;
}

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA/HIP");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void k3_v3_ll_combine(
    const torch::Tensor& output_workspace,
    const torch::Tensor& act_fp8,
    const torch::Tensor& act_scale,
    const torch::Tensor& m_indices,
    const torch::Tensor& weight_pack5,
    const torch::Tensor& weight_scale,
    const torch::Tensor& row_combine_ptrs,
    const bool tail_reduce = false,
    const int64_t ll_block_m = 32) {
    check_cuda_contiguous(output_workspace, "output_workspace");
    check_cuda_contiguous(act_fp8, "act_fp8");
    check_cuda_contiguous(act_scale, "act_scale");
    check_cuda_contiguous(m_indices, "m_indices");
    check_cuda_contiguous(weight_pack5, "weight_pack5");
    check_cuda_contiguous(weight_scale, "weight_scale");
    check_cuda_contiguous(row_combine_ptrs, "row_combine_ptrs");
    TORCH_CHECK(!tail_reduce,
                "V3 K3 LL pack5 tail_reduce uses k3_v3_ll_combine_tail");
    TORCH_CHECK(output_workspace.scalar_type() == torch::kBFloat16,
                "output_workspace must be BF16");
    TORCH_CHECK(act_fp8.scalar_type() == torch::kFloat8_e4m3fn,
                "act_fp8 must be FP8 E4M3");
    TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32,
                "act_scale must be FP32");
    TORCH_CHECK(m_indices.scalar_type() == torch::kInt,
                "m_indices must be int32");
    TORCH_CHECK(weight_pack5.scalar_type() == torch::kFloat8_e4m3fn,
                "weight_pack5 must be FP8 E4M3");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32,
                "weight_scale must be FP32");
    TORCH_CHECK(row_combine_ptrs.scalar_type() == torch::kInt64,
                "row_combine_ptrs must be int64");
    TORCH_CHECK(output_workspace.dim() == 2,
                "output_workspace must be [rows, hidden]");
    TORCH_CHECK(act_fp8.dim() == 2, "act_fp8 must be [rows, intermediate]");
    TORCH_CHECK(output_workspace.size(0) == act_fp8.size(0),
                "output_workspace and act_fp8 row counts must match");
    const int total_rows = static_cast<int>(act_fp8.size(0));
    const int intermediate = static_cast<int>(act_fp8.size(1));
    const int hidden = static_cast<int>(output_workspace.size(1));
    TORCH_CHECK(total_rows > 0 && total_rows % 64 == 0,
                "V3 K3 LL pack5 expects rows padded to 64");
    TORCH_CHECK(hidden == 4096 && intermediate == 2048,
                "V3 K3 LL pack5 is specialized for hidden=4096, intermediate=2048");
    TORCH_CHECK(ll_block_m == 32 || ll_block_m == 48 || ll_block_m == 64,
                "V3 K3 LL pack5 expects ll_block_m in {32, 48, 64}");
    TORCH_CHECK(weight_scale.dim() == 2 && weight_scale.size(1) == hidden,
                "weight_scale must be [local_experts, hidden]");
    TORCH_CHECK(act_scale.numel() >= total_rows,
                "act_scale must have at least one scale per row");
    TORCH_CHECK(m_indices.numel() >= weight_scale.size(0),
                "m_indices must carry one actual row count per local expert");
    TORCH_CHECK(row_combine_ptrs.numel() >= total_rows,
                "row_combine_ptrs length must cover total rows");
    TORCH_CHECK(weight_pack5.dim() >= 1 &&
                    weight_pack5.numel() >=
                        weight_scale.size(0) * static_cast<int64_t>(hidden) *
                            static_cast<int64_t>(intermediate),
                "weight_pack5 must contain flattened [local_experts, hidden, intermediate]");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dcu_megamoe_v3_launch_k3_ll_combine_pack5(
        reinterpret_cast<hip_bfloat16*>(output_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(act_fp8.data_ptr()),
        act_scale.data_ptr<float>(),
        m_indices.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(weight_pack5.data_ptr()),
        weight_scale.data_ptr<float>(),
        row_combine_ptrs.data_ptr<int64_t>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        total_rows,
        0,
        static_cast<int>(weight_scale.size(0)),
        0,
        0,
        nullptr,
        0,
        hidden,
        0,
        static_cast<int>(ll_block_m),
        nullptr,
        stream);
    const hipError_t post_launch_status = hipGetLastError();
    TORCH_CHECK(post_launch_status == hipSuccess,
                "hipGetLastError after V3 K3 LL pack5 launch failed: ",
                hipGetErrorString(post_launch_status));
}

void k3_v3_ll_combine_tail(
    const torch::Tensor& output_workspace,
    const torch::Tensor& act_fp8,
    const torch::Tensor& act_scale,
    const torch::Tensor& m_indices,
    const torch::Tensor& weight_pack5,
    const torch::Tensor& weight_scale,
    const torch::Tensor& row_combine_ptrs,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& done_counter,
    const torch::Tensor& signal_addrs,
    const torch::Tensor& reduce_y,
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t num_topk,
    const int64_t ll_block_m = 32,
    const std::optional<torch::Tensor>& signal_generation_tensor = std::nullopt,
    const std::optional<torch::Tensor>& runtime_num_tokens_tensor = std::nullopt) {
    check_cuda_contiguous(output_workspace, "output_workspace");
    check_cuda_contiguous(act_fp8, "act_fp8");
    check_cuda_contiguous(act_scale, "act_scale");
    check_cuda_contiguous(m_indices, "m_indices");
    check_cuda_contiguous(weight_pack5, "weight_pack5");
    check_cuda_contiguous(weight_scale, "weight_scale");
    check_cuda_contiguous(row_combine_ptrs, "row_combine_ptrs");
    check_cuda_contiguous(sym_buffer, "sym_buffer");
    check_cuda_contiguous(done_counter, "done_counter");
    check_cuda_contiguous(signal_addrs, "signal_addrs");
    check_cuda_contiguous(reduce_y, "reduce_y");
    TORCH_CHECK(output_workspace.scalar_type() == torch::kBFloat16,
                "output_workspace must be BF16");
    TORCH_CHECK(act_fp8.scalar_type() == torch::kFloat8_e4m3fn,
                "act_fp8 must be FP8 E4M3");
    TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32,
                "act_scale must be FP32");
    TORCH_CHECK(m_indices.scalar_type() == torch::kInt,
                "m_indices must be int32");
    TORCH_CHECK(weight_pack5.scalar_type() == torch::kFloat8_e4m3fn,
                "weight_pack5 must be FP8 E4M3");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32,
                "weight_scale must be FP32");
    TORCH_CHECK(row_combine_ptrs.scalar_type() == torch::kInt64,
                "row_combine_ptrs must be int64");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kUInt8 ||
                    sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be uint8/int8");
    TORCH_CHECK(done_counter.scalar_type() == torch::kInt,
                "done_counter must be int32");
    TORCH_CHECK(signal_addrs.scalar_type() == torch::kInt64,
                "signal_addrs must be int64");
    TORCH_CHECK(reduce_y.scalar_type() == torch::kBFloat16,
                "reduce_y must be BF16");
    TORCH_CHECK(output_workspace.dim() == 2,
                "output_workspace must be [rows, hidden]");
    TORCH_CHECK(act_fp8.dim() == 2,
                "act_fp8 must be [rows, intermediate]");
    const int total_rows = static_cast<int>(act_fp8.size(0));
    const int intermediate = static_cast<int>(act_fp8.size(1));
    const int hidden = static_cast<int>(output_workspace.size(1));
    TORCH_CHECK(output_workspace.size(0) == total_rows,
                "output_workspace and act_fp8 row counts must match");
    TORCH_CHECK(reduce_y.dim() == 2 &&
                    reduce_y.size(0) == num_tokens &&
                    reduce_y.size(1) == hidden,
                "reduce_y shape mismatch");
    TORCH_CHECK(total_rows > 0 && total_rows % 64 == 0,
                "V3 K3 LL pack5 expects rows padded to 64");
    TORCH_CHECK(hidden == 4096 && intermediate == 2048,
                "V3 K3 LL pack5 is specialized for hidden=4096, intermediate=2048");
    TORCH_CHECK(ll_block_m == 32 || ll_block_m == 48 || ll_block_m == 64,
                "V3 K3 LL pack5 expects ll_block_m in {32, 48, 64}");
    TORCH_CHECK(num_ranks > 0 && num_ranks <= 8,
                "num_ranks must be in [1, 8]");
    TORCH_CHECK(num_experts > 0 && num_experts % num_ranks == 0,
                "num_experts must be divisible by num_ranks");
    TORCH_CHECK(num_topk > 0 && num_tokens >= 0,
                "num_topk must be positive and num_tokens must be non-negative");
    TORCH_CHECK(num_max_tokens_per_rank >= num_tokens,
                "num_max_tokens_per_rank must cover num_tokens");
    constexpr int64_t kTailDoneCounterRingSlots = 16;
    constexpr int64_t kTailDoneCounterInts = 3 * kTailDoneCounterRingSlots;
    TORCH_CHECK(done_counter.numel() >= kTailDoneCounterInts,
                "done_counter must cover LL tail done and peer-ready rings");
    TORCH_CHECK(signal_addrs.numel() >= 16,
                "signal_addrs must have 16 entries");
    const int32_t* signal_generation_ptr = nullptr;
    if (signal_generation_tensor.has_value()) {
        const auto& generation = signal_generation_tensor.value();
        check_cuda_contiguous(generation, "signal_generation_tensor");
        TORCH_CHECK(generation.scalar_type() == torch::kInt &&
                        generation.numel() >= 1,
                    "signal_generation_tensor must be a CUDA int32 tensor");
        signal_generation_ptr = generation.data_ptr<int32_t>();
    }
    const int32_t* runtime_num_tokens_ptr = nullptr;
    if (runtime_num_tokens_tensor.has_value()) {
        const auto& runtime = runtime_num_tokens_tensor.value();
        check_cuda_contiguous(runtime, "runtime_num_tokens_tensor");
        TORCH_CHECK(runtime.scalar_type() == torch::kInt &&
                        runtime.numel() >= 1,
                    "runtime_num_tokens_tensor must be a CUDA int32 tensor");
        runtime_num_tokens_ptr = runtime.data_ptr<int32_t>();
    }
    const int local_experts = num_ranks > 0 ? (num_experts / num_ranks) : 0;
    TORCH_CHECK(m_indices.numel() >= local_experts,
                "m_indices must carry one actual row count per local expert");
    TORCH_CHECK(row_combine_ptrs.numel() >= total_rows,
                "row_combine_ptrs length must cover total rows");
    TORCH_CHECK(weight_scale.dim() == 2 && weight_scale.size(1) == hidden,
                "weight_scale must be [local_experts, hidden]");
    TORCH_CHECK(weight_pack5.dim() >= 1 &&
                    weight_pack5.numel() >=
                        weight_scale.size(0) * static_cast<int64_t>(hidden) *
                            static_cast<int64_t>(intermediate),
                "weight_pack5 must contain flattened [local_experts, hidden, intermediate]");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dcu_megamoe_v3_launch_k3_ll_combine_pack5(
        reinterpret_cast<hip_bfloat16*>(output_workspace.data_ptr()),
        reinterpret_cast<const uint8_t*>(act_fp8.data_ptr()),
        act_scale.data_ptr<float>(),
        m_indices.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(weight_pack5.data_ptr()),
        weight_scale.data_ptr<float>(),
        row_combine_ptrs.data_ptr<int64_t>(),
        reinterpret_cast<uint8_t*>(sym_buffer.data_ptr()),
        done_counter.data_ptr<int32_t>(),
        signal_addrs.data_ptr<int64_t>(),
        reinterpret_cast<hip_bfloat16*>(reduce_y.data_ptr()),
        total_rows,
        static_cast<int>(num_ranks),
        static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_tokens),
        runtime_num_tokens_ptr,
        static_cast<int>(num_topk),
        hidden,
        1,
        static_cast<int>(ll_block_m),
        signal_generation_ptr,
        stream);
    const hipError_t post_launch_status = hipGetLastError();
    TORCH_CHECK(post_launch_status == hipSuccess,
                "hipGetLastError after V3 K3 LL tail launch failed: ",
                hipGetErrorString(post_launch_status));
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k3_v3_ll_combine",
          &k3_v3_ll_combine,
          pybind11::arg("output_workspace"),
          pybind11::arg("act_fp8"),
          pybind11::arg("act_scale"),
          pybind11::arg("m_indices"),
          pybind11::arg("weight_pack5"),
          pybind11::arg("weight_scale"),
          pybind11::arg("row_combine_ptrs"),
          pybind11::arg("tail_reduce") = false,
          pybind11::arg("ll_block_m") = 32);
    m.def("k3_v3_ll_combine_tail",
          &k3_v3_ll_combine_tail,
          pybind11::arg("output_workspace"),
          pybind11::arg("act_fp8"),
          pybind11::arg("act_scale"),
          pybind11::arg("m_indices"),
          pybind11::arg("weight_pack5"),
          pybind11::arg("weight_scale"),
          pybind11::arg("row_combine_ptrs"),
          pybind11::arg("sym_buffer"),
          pybind11::arg("done_counter"),
          pybind11::arg("signal_addrs"),
          pybind11::arg("reduce_y"),
          pybind11::arg("num_ranks"),
          pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("ll_block_m") = 32,
          pybind11::arg("signal_generation_tensor") = std::nullopt,
          pybind11::arg("runtime_num_tokens_tensor") = std::nullopt);
}
