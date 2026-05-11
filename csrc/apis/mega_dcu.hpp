#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/python.h>

#include <deep_gemm/layout/mega_moe_dcu.cuh>

namespace deep_gemm::mega {

static constexpr int kMaxCandidateBlockM = 192;

static int get_token_alignment_for_mega_moe() {
    return kTokenAlignment;
}

static pybind11::dict get_mega_moe_hip_build_config() {
    pybind11::dict config;
    config["operator"] = "fp8_w8a8_mega_moe";
    config["package"] = "megamoe";
    config["weight_scale_mode"] = "channelwise_fp32";
    config["input_scale_mode"] = "channelwise_fp32";
    config["l2_act_scale_mode"] = "channelwise_fp32";
    config["has_fp8_api"] = true;
    config["has_fp4_api"] = false;
    config["fusion_boundary"] = "dispatch_pool_l1_swiglu_quant_l2_combine";
    return config;
}

void launch_mega_moe_multirank_persistent_hip_w8a8_channelwise(
    void* y,
    const void* l1_weights, const float* l1_weights_sf,
    const void* l2_weights, const float* l2_weights_sf,
    int* cumulative_local_expert_recv_stats,
    const int64_t* sym_buffer_ptrs,
    void* route_scratch,
    int rank_idx, int num_ranks,
    int num_max_tokens_per_rank,
    int num_experts_per_rank,
    int num_tokens, int num_topk,
    int hidden, int intermediate_hidden,
    float activation_clamp,
    bool fast_math);

void launch_mega_moe_deepep_scatter_channelwise_hip(
    void* grouped_x,
    float* grouped_x_scale,
    float* route_weights,
    int* m_indices,
    int* output_index,
    int* expert_start_loc,
    const void* recv_x,
    const float* recv_x_scale,
    const void* recv_topk_ids,
    const float* recv_topk_weights,
    const int* num_recv_tokens_per_expert,
    int total_rows,
    int recv_rows,
    int topk,
    int hidden,
    int num_experts,
    bool topk_ids_i64);

void launch_mega_moe_deepep_gather_channelwise_hip(
    void* recv_y,
    const void* l2_out,
    const void* recv_topk_ids,
    const float* recv_topk_weights,
    const int* output_index,
    int recv_rows,
    int topk,
    int hidden,
    bool topk_ids_i64,
    bool apply_topk_weights);

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool&, const std::string&) {
    TORCH_CHECK(num_ranks > 0, "num_ranks must be positive");
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    TORCH_CHECK(hidden > 0 && intermediate_hidden > 0, "hidden sizes must be positive");

    const int64_t input_token_offset = workspace_bytes(num_ranks, num_experts, num_max_tokens_per_rank);
    const int64_t input_sf_offset =
        input_token_offset + static_cast<int64_t>(num_max_tokens_per_rank) * hidden;
    const int64_t topk_idx_offset =
        input_sf_offset + static_cast<int64_t>(num_max_tokens_per_rank) * sizeof(float);
    const int64_t topk_weights_offset =
        topk_idx_offset + static_cast<int64_t>(num_max_tokens_per_rank) * num_topk * sizeof(int64_t);
    const int64_t combine_offset =
        topk_weights_offset + static_cast<int64_t>(num_max_tokens_per_rank) * num_topk * sizeof(float);
    const int64_t total_bytes = align_i64(
        combine_offset +
            static_cast<int64_t>(num_topk) * num_max_tokens_per_rank *
                hidden * static_cast<int64_t>(sizeof(uint16_t)),
        16);

    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto* byte_base = static_cast<uint8_t*>(buffer.data_ptr());
        const auto device = buffer.device();
        const auto fp8_options = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);
        const auto f32_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        const auto i64_options = torch::TensorOptions().dtype(torch::kInt64).device(device);

        auto x = torch::from_blob(
            byte_base + input_token_offset,
            {num_max_tokens_per_rank, hidden},
            fp8_options);
        auto x_sf = torch::from_blob(
            byte_base + input_sf_offset,
            {num_max_tokens_per_rank},
            f32_options);
        auto topk_idx = torch::from_blob(
            byte_base + topk_idx_offset,
            {num_max_tokens_per_rank, num_topk},
            i64_options);
        auto topk_weights = torch::from_blob(
            byte_base + topk_weights_offset,
            {num_max_tokens_per_rank, num_topk},
            f32_options);
        auto empty_fp8 = torch::from_blob(byte_base + combine_offset, {0}, fp8_options);
        auto empty_f32 = torch::from_blob(byte_base + combine_offset, {0}, f32_options);
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               empty_fp8, empty_f32, empty_fp8, empty_f32);
    };
    return {total_bytes, slice_input_buffers};
}

static int64_t get_mega_moe_route_scratch_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool&, const std::string&) {
    TORCH_CHECK(num_ranks > 0, "num_ranks must be positive");
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    TORCH_CHECK(hidden > 0 && intermediate_hidden > 0, "hidden sizes must be positive");

    constexpr int route_tile_m = 1 << kDcuRouteTileMMinLog2;
    return dcu_route_scratch_bytes(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        route_tile_m, hidden, intermediate_hidden);
}

static void set_mega_moe_peer_ptrs(
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const std::vector<int64_t>& signal_ptrs) {
    static_assert(sizeof(int64_t) == sizeof(uint8_t*),
                  "DCU peer pointer header expects 64-bit device pointers");
    static_assert(sizeof(int64_t) == sizeof(int*),
                  "DCU signal pointer header expects 64-bit device pointers");

    TORCH_CHECK(sym_buffer.is_cuda(), "sym_buffer must be CUDA/HIP memory");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8, "sym_buffer must be int8");
    TORCH_CHECK(sym_buffer.is_contiguous(), "sym_buffer must be contiguous");
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    TORCH_CHECK(num_ranks > 0, "sym_buffer_ptrs must be non-empty");
    TORCH_CHECK(signal_ptrs.size() == sym_buffer_ptrs.size(),
                "signal_ptrs must match sym_buffer_ptrs");
    TORCH_CHECK(static_cast<int64_t>(sym_buffer.nbytes()) >= dcu_workspace_offset(num_ranks),
                "sym_buffer is too small for DCU MegaMoE peer pointer header");

    auto* base = static_cast<uint8_t*>(sym_buffer.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipError_t status = hipMemcpyAsync(
        base + dcu_sym_buffer_ptrs_offset(),
        sym_buffer_ptrs.data(),
        sizeof(int64_t) * num_ranks,
        hipMemcpyHostToDevice,
        stream);
    TORCH_CHECK(status == hipSuccess,
                "copying DCU MegaMoE peer sym_buffer pointers failed: ",
                hipGetErrorString(status));
    status = hipMemcpyAsync(
        base + dcu_signal_ptrs_offset(num_ranks),
        signal_ptrs.data(),
        sizeof(int64_t) * num_ranks,
        hipMemcpyHostToDevice,
        stream);
    TORCH_CHECK(status == hipSuccess,
                "copying DCU MegaMoE peer signal pointers failed: ",
                hipGetErrorString(status));
}

static torch::Tensor transform_sf_into_required_layout(
    const torch::Tensor& sf,
    const int&, const int&,
    const std::tuple<int, int, int>&,
    const std::optional<int>& = std::nullopt,
    const std::optional<bool>& = std::nullopt,
    const bool& = false) {
    return sf.contiguous();
}

static pybind11::tuple deepep_deepgemm_preprocess_channelwise(
    const torch::Tensor& recv_x,
    const torch::Tensor& recv_x_scale,
    const torch::Tensor& recv_topk_ids,
    const torch::Tensor& recv_topk_weights,
    const torch::Tensor& num_recv_tokens_per_expert,
    const int& all_tokens) {
    TORCH_CHECK(recv_x.is_cuda() && recv_x_scale.is_cuda() && recv_topk_ids.is_cuda() &&
                recv_topk_weights.is_cuda() && num_recv_tokens_per_expert.is_cuda(),
                "DeepEP preprocess tensors must be CUDA/HIP tensors");
    TORCH_CHECK(recv_x.scalar_type() == torch::kFloat8_e4m3fn,
                "DeepEP preprocess expects FP8 E4M3 recv_x");
    TORCH_CHECK(recv_x_scale.scalar_type() == torch::kFloat32 &&
                recv_topk_weights.scalar_type() == torch::kFloat32,
                "DeepEP preprocess scales and weights must be FP32");
    TORCH_CHECK(recv_topk_ids.scalar_type() == torch::kLong ||
                recv_topk_ids.scalar_type() == torch::kInt,
                "DeepEP preprocess topk ids must be int64 or int32");
    TORCH_CHECK(num_recv_tokens_per_expert.scalar_type() == torch::kInt,
                "num_recv_tokens_per_expert must be int32");
    TORCH_CHECK(recv_x.is_contiguous() && recv_x_scale.is_contiguous() &&
                recv_topk_ids.is_contiguous() && recv_topk_weights.is_contiguous() &&
                num_recv_tokens_per_expert.is_contiguous(),
                "DeepEP preprocess tensors must be contiguous");
    TORCH_CHECK(recv_x.dim() == 2, "recv_x must be [tokens, hidden]");
    TORCH_CHECK(recv_topk_ids.dim() == 2 && recv_topk_weights.dim() == 2,
                "topk ids/weights must be [tokens, topk]");
    TORCH_CHECK(recv_topk_ids.sizes() == recv_topk_weights.sizes(),
                "topk ids/weights shape mismatch");
    TORCH_CHECK(recv_topk_ids.size(0) == recv_x.size(0), "topk rows must match recv_x");
    TORCH_CHECK(recv_x_scale.numel() == recv_x.size(0),
                "channelwise recv_x_scale must have one value per received token");
    TORCH_CHECK(all_tokens >= 0, "all_tokens must be non-negative");

    const int rows = std::max(all_tokens, 1);
    const int hidden = static_cast<int>(recv_x.size(1));
    const int recv_rows = static_cast<int>(recv_x.size(0));
    const int topk = static_cast<int>(recv_topk_ids.size(1));
    const int num_experts = static_cast<int>(num_recv_tokens_per_expert.numel());
    const auto device = recv_x.device();
    const auto fp8_options = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);
    const auto f32_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto i32_options = torch::TensorOptions().dtype(torch::kInt).device(device);

    auto grouped_x = torch::empty({rows, hidden}, fp8_options);
    auto grouped_x_scale = torch::empty({rows}, f32_options);
    auto route_weights = torch::zeros({rows}, f32_options);
    auto m_indices = torch::full({rows}, -1, i32_options);
    auto output_index = torch::full({recv_rows, topk}, -1, i32_options);
    auto expert_start_loc = torch::empty({num_experts}, i32_options);

    launch_mega_moe_deepep_scatter_channelwise_hip(
        grouped_x.data_ptr(),
        grouped_x_scale.data_ptr<float>(),
        route_weights.data_ptr<float>(),
        m_indices.data_ptr<int>(),
        output_index.data_ptr<int>(),
        expert_start_loc.data_ptr<int>(),
        recv_x.data_ptr(),
        recv_x_scale.data_ptr<float>(),
        recv_topk_ids.data_ptr(),
        recv_topk_weights.data_ptr<float>(),
        num_recv_tokens_per_expert.data_ptr<int>(),
        rows,
        recv_rows,
        topk,
        hidden,
        num_experts,
        recv_topk_ids.scalar_type() == torch::kLong);
    return pybind11::make_tuple(grouped_x, grouped_x_scale, route_weights, m_indices, output_index);
}

static void deepep_deepgemm_postprocess_channelwise(
    const torch::Tensor& recv_y,
    const torch::Tensor& l2_out,
    const torch::Tensor& recv_topk_ids,
    const torch::Tensor& recv_topk_weights,
    const torch::Tensor& output_index,
    const bool& apply_topk_weights) {
    TORCH_CHECK(recv_y.is_cuda() && l2_out.is_cuda() && recv_topk_ids.is_cuda() &&
                recv_topk_weights.is_cuda() && output_index.is_cuda(),
                "DeepEP postprocess tensors must be CUDA/HIP tensors");
    TORCH_CHECK(recv_y.scalar_type() == torch::kBFloat16 && l2_out.scalar_type() == torch::kBFloat16,
                "DeepEP postprocess expects BF16 tensors");
    TORCH_CHECK(recv_topk_ids.scalar_type() == torch::kLong ||
                recv_topk_ids.scalar_type() == torch::kInt,
                "DeepEP postprocess topk ids must be int64 or int32");
    TORCH_CHECK(recv_topk_weights.scalar_type() == torch::kFloat32,
                "DeepEP postprocess topk weights must be FP32");
    TORCH_CHECK(output_index.scalar_type() == torch::kInt, "output_index must be int32");
    TORCH_CHECK(recv_y.is_contiguous() && l2_out.is_contiguous() &&
                recv_topk_ids.is_contiguous() && recv_topk_weights.is_contiguous() &&
                output_index.is_contiguous(),
                "DeepEP postprocess tensors must be contiguous");
    TORCH_CHECK(recv_y.dim() == 2 && l2_out.dim() == 2, "recv_y/l2_out must be 2D");
    TORCH_CHECK(recv_topk_ids.dim() == 2 && recv_topk_weights.dim() == 2 &&
                output_index.dim() == 2,
                "topk ids/weights/output_index must be 2D");
    TORCH_CHECK(recv_topk_ids.sizes() == recv_topk_weights.sizes() &&
                recv_topk_ids.sizes() == output_index.sizes(),
                "topk ids/weights/output_index shape mismatch");
    TORCH_CHECK(recv_y.size(0) == recv_topk_ids.size(0), "recv_y rows mismatch");
    TORCH_CHECK(recv_y.size(1) == l2_out.size(1), "hidden size mismatch");
    launch_mega_moe_deepep_gather_channelwise_hip(
        recv_y.data_ptr(),
        l2_out.data_ptr(),
        recv_topk_ids.data_ptr(),
        recv_topk_weights.data_ptr<float>(),
        output_index.data_ptr<int>(),
        static_cast<int>(recv_y.size(0)),
        static_cast<int>(recv_topk_ids.size(1)),
        static_cast<int>(recv_y.size(1)),
        recv_topk_ids.scalar_type() == torch::kLong,
        apply_topk_weights);
}

static void fp8_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& route_scratch,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const std::vector<int64_t>& signal_ptrs,
    const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math) {
    const auto [rm, rn, rk] = recipe;
    TORCH_CHECK(rm == 1 && rn == 1 && rk == 32, "DCU W8A8 MegaMoE expects recipe=(1,1,32)");
    TORCH_CHECK(activation == "swiglu", "DCU W8A8 MegaMoE supports swiglu only");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "y must be BF16");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8, "sym_buffer must be int8");
    TORCH_CHECK(route_scratch.is_cuda(), "route_scratch must be CUDA/HIP memory");
    TORCH_CHECK(route_scratch.scalar_type() == torch::kInt8, "route_scratch must be int8");
    TORCH_CHECK(route_scratch.is_contiguous(), "route_scratch must be contiguous");

    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;
    TORCH_CHECK(l1_weights.scalar_type() == torch::kFloat8_e4m3fn,
                "DCU MegaMoE expects FP8 E4M3 L1 weights");
    TORCH_CHECK(l2_weights.scalar_type() == torch::kFloat8_e4m3fn,
                "DCU MegaMoE expects FP8 E4M3 L2 weights");
    TORCH_CHECK(l1_weights_sf.scalar_type() == torch::kFloat32 &&
                l2_weights_sf.scalar_type() == torch::kFloat32,
                "DCU MegaMoE expects FP32 channelwise weight scales");
    TORCH_CHECK(l1_weights.dim() == 3 && l2_weights.dim() == 3, "weights must be grouped 3D tensors");
    TORCH_CHECK(l1_weights_sf.dim() == 2 && l2_weights_sf.dim() == 2, "weight scales must be [expert,row]");

    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    TORCH_CHECK(signal_ptrs.size() == sym_buffer_ptrs.size(), "signal_ptrs must match sym_buffer_ptrs");
    const int num_experts_per_rank = static_cast<int>(l1_weights.size(0));
    const int num_tokens = static_cast<int>(y.size(0));
    const int hidden = static_cast<int>(y.size(1));
    TORCH_CHECK(l1_weights_sf.size(1) % 2 == 0, "invalid L1 scale shape");
    const int intermediate_hidden = static_cast<int>(l1_weights_sf.size(1) / 2);
    const int l1_rows = intermediate_hidden * 2;
    const int l2_rows = hidden;

    TORCH_CHECK(num_ranks > 0, "invalid num_ranks");
    TORCH_CHECK(num_experts == num_experts_per_rank * num_ranks, "invalid expert count");
    TORCH_CHECK(l1_rows % 16 == 0 && hidden % 16 == 0 && l2_rows % 16 == 0 &&
                intermediate_hidden % 16 == 0,
                "DCU MegaMoE Marlin weights require rows and K divisible by 16");
    TORCH_CHECK(l1_weights.size(1) == l1_rows / 16 &&
                l1_weights.size(2) == hidden * 16,
                "invalid Marlin L1 weight shape, expected [expert, rows/16, K*16]");
    TORCH_CHECK(l2_weights.size(1) == l2_rows / 16 &&
                l2_weights.size(2) == intermediate_hidden * 16,
                "invalid Marlin L2 weight shape, expected [expert, rows/16, K*16]");
    TORCH_CHECK(l1_weights_sf.size(0) == num_experts_per_rank &&
                l1_weights_sf.size(1) == l1_rows,
                "invalid L1 scale shape");
    TORCH_CHECK(l2_weights_sf.size(0) == num_experts_per_rank &&
                l2_weights_sf.size(1) == l2_rows,
                "invalid L2 scale shape");
    TORCH_CHECK(num_tokens <= num_max_tokens_per_rank, "too many tokens");
    const int64_t num_required_bytes = std::get<0>(get_symm_buffer_size_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, true, activation));
    TORCH_CHECK(static_cast<int64_t>(sym_buffer.nbytes()) >= num_required_bytes,
                "sym_buffer is too small for the requested DCU MegaMoE capacity");
    const int64_t route_scratch_required_bytes = get_mega_moe_route_scratch_size_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden, true, activation);
    TORCH_CHECK(static_cast<int64_t>(route_scratch.nbytes()) >= route_scratch_required_bytes,
                "route_scratch is too small for the requested DCU MegaMoE capacity");

    int* stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value()) {
        TORCH_CHECK(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt,
                    "stats must be int32");
        TORCH_CHECK(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank,
                    "stats must have num_experts_per_rank elements");
        stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
    }

    const float activation_clamp = activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    launch_mega_moe_multirank_persistent_hip_w8a8_channelwise(
        y.data_ptr(),
        l1_weights.data_ptr(), l1_weights_sf.data_ptr<float>(),
        l2_weights.data_ptr(), l2_weights_sf.data_ptr<float>(),
        stats_ptr, sym_buffer_ptrs.data(), route_scratch.data_ptr(),
        rank_idx, num_ranks, num_max_tokens_per_rank,
        num_experts_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, activation_clamp, fast_math);
}

static void register_apis(pybind11::module_& m) {
    m.def("get_token_alignment_for_mega_moe", &get_token_alignment_for_mega_moe);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
    m.def("get_mega_moe_route_scratch_size_for_mega_moe",
          &get_mega_moe_route_scratch_size_for_mega_moe);
    m.def("get_mega_moe_hip_build_config", &get_mega_moe_hip_build_config);
    m.def("set_mega_moe_peer_ptrs", &set_mega_moe_peer_ptrs);
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout,
          pybind11::arg("sf"), pybind11::arg("mn"), pybind11::arg("k"), pybind11::arg("recipe"),
          pybind11::arg("num_groups") = std::nullopt,
          pybind11::arg("is_sfa") = std::nullopt,
          pybind11::arg("disable_ue8m0_cast") = false);
    m.def("fp8_mega_moe", &fp8_mega_moe);
    m.def("fp8_w8a8_mega_moe", &fp8_mega_moe);
    m.def("deepep_deepgemm_preprocess_channelwise", &deepep_deepgemm_preprocess_channelwise,
          pybind11::arg("recv_x"),
          pybind11::arg("recv_x_scale"),
          pybind11::arg("recv_topk_ids"),
          pybind11::arg("recv_topk_weights"),
          pybind11::arg("num_recv_tokens_per_expert"),
          pybind11::arg("all_tokens"));
    m.def("deepep_deepgemm_postprocess_channelwise", &deepep_deepgemm_postprocess_channelwise,
          pybind11::arg("recv_y"),
          pybind11::arg("l2_out"),
          pybind11::arg("recv_topk_ids"),
          pybind11::arg("recv_topk_weights"),
          pybind11::arg("output_index"),
          pybind11::arg("apply_topk_weights") = false);
}

} // namespace deep_gemm::mega
