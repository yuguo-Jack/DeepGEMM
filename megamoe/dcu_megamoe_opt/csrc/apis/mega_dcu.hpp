#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <torch/python.h>

#include <mega_moe_dcu/layout.cuh>

namespace deep_gemm::mega {

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

using MegaMoeSymmBufferSlices = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor>;

static std::tuple<int64_t, std::function<MegaMoeSymmBufferSlices(const torch::Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool&, const std::string&) {
    TORCH_CHECK(num_ranks > 0, "num_ranks must be positive");
    TORCH_CHECK(num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks");
    TORCH_CHECK(hidden > 0 && intermediate_hidden > 0, "hidden sizes must be positive");

    const int64_t input_token_offset = workspace_bytes(num_ranks, num_experts, num_max_tokens_per_rank);
    const int64_t runtime_num_tokens_offset = dcu_runtime_num_tokens_offset(num_ranks);
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
        const auto i32_options = torch::TensorOptions().dtype(torch::kInt).device(device);

        auto runtime_num_tokens = torch::from_blob(
            byte_base + runtime_num_tokens_offset,
            {1},
            i32_options);
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
                               empty_fp8, empty_f32, empty_fp8, empty_f32,
                               runtime_num_tokens);
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

    const bool staged_pack5_shape =
        num_ranks == 8 && num_experts == 256 && num_topk == 6 &&
        hidden == 4096 && intermediate_hidden == 2048;
    TORCH_CHECK(
        staged_pack5_shape,
        "DCU MegaMoE staged LL/normal route_scratch currently supports only "
        "EP8 experts=256 topk=6 hidden=4096 intermediate=2048");

    constexpr int64_t kK1RouteTileM = 256;
    constexpr int64_t kK1Alignment = 256;
    constexpr int64_t kK1RouteCapacitySlack = 64;
    constexpr int64_t kK1RouteCapacitySlackDivisor = 10;
    constexpr int64_t kK1LlRowTile = 64;
    constexpr int64_t kK1LlHeadroomExpectedRowsThreshold = 48;
    constexpr int64_t kK1LlHeadroomRows = 64;
    constexpr int64_t kK1AsmLaunchArgsBytes = 256;
    const auto ceil_div_i64 = [](const int64_t value, const int64_t divisor) {
        return (value + divisor - 1) / divisor;
    };
    const auto route_capacity_headroom_rows =
        [&](const int64_t expected_per_expert) {
            return std::max<int64_t>(
                kK1RouteCapacitySlack,
                ceil_div_i64(expected_per_expert,
                             kK1RouteCapacitySlackDivisor));
        };
    const int64_t local_experts = num_experts / num_ranks;
    const int64_t ll_expected_rows_per_expert = std::max<int64_t>(
        1, ceil_div_i64(
               static_cast<int64_t>(num_max_tokens_per_rank) * num_topk,
               local_experts));
    int64_t ll_rows_per_expert =
        align_i64(ll_expected_rows_per_expert, kK1LlRowTile);
    const int64_t min_slack =
        ll_expected_rows_per_expert >= kK1LlHeadroomExpectedRowsThreshold
            ? kK1LlHeadroomRows
            : 0;
    if (ll_rows_per_expert - ll_expected_rows_per_expert < min_slack) {
        ll_rows_per_expert = align_i64(
            ll_expected_rows_per_expert + min_slack, kK1LlRowTile);
    }
    const int64_t ll_capacity_rows = local_experts * ll_rows_per_expert;
    const int64_t total_tasks =
        static_cast<int64_t>(num_ranks) * num_max_tokens_per_rank * num_topk;
    const int64_t expected_per_expert =
        ceil_div_i64(total_tasks, num_experts);
    const int64_t rows_per_expert_target = std::max<int64_t>(
        kK1Alignment,
        expected_per_expert +
            route_capacity_headroom_rows(expected_per_expert));
    const int64_t fixed_capacity_tiles_per_expert =
        ceil_div_i64(rows_per_expert_target, kK1RouteTileM);
    const int64_t normal_capacity_rows =
        local_experts * fixed_capacity_tiles_per_expert * kK1RouteTileM;
    const int64_t capacity_rows = std::max(ll_capacity_rows, normal_capacity_rows);

    int64_t offset = 0;
    offset += capacity_rows * static_cast<int64_t>(hidden);
    offset = align_i64(offset, 16);
    const int64_t k1_row_tiles = ceil_div_i64(capacity_rows, kK1RouteTileM);
    const int64_t k1_l1_tiles =
        ceil_div_i64(static_cast<int64_t>(intermediate_hidden) * 2, kK1RouteTileM);
    offset += k1_row_tiles * k1_l1_tiles * static_cast<int64_t>(sizeof(int32_t));
    offset = align_i64(offset, 16);
    offset += k1_row_tiles * static_cast<int64_t>(sizeof(int32_t));
    offset = align_i64(offset, 16);
    offset += kK1AsmLaunchArgsBytes;
    offset = align_i64(offset, 16);
    offset += capacity_rows * static_cast<int64_t>(intermediate_hidden) * 2 *
              static_cast<int64_t>(sizeof(uint16_t));
    offset = align_i64(offset, 16);
    offset += capacity_rows * static_cast<int64_t>(intermediate_hidden);
    offset = align_i64(offset, 16);
    offset += capacity_rows * static_cast<int64_t>(sizeof(float));
    offset = align_i64(offset, 16);

    constexpr int64_t kProbStorageBytes = 256;
    constexpr int64_t kTailDoneCounterRingSlots = 16;
    constexpr int64_t kTailDoneCounterInts = 80;
    static_assert(kTailDoneCounterInts == 3 * kTailDoneCounterRingSlots + 32,
                  "tail done counter scratch layout changed");
    constexpr int64_t kTailSignalAddrs = 16;
    const int64_t route_base =
        route_task_workspace_bytes(num_ranks, num_experts, num_max_tokens_per_rank);
    const int64_t prob_offset = route_base + offset;
    const int64_t tail_done_offset =
        align_i64(prob_offset + kProbStorageBytes, sizeof(int32_t));
    const int64_t tail_signal_addrs_offset = align_i64(
        tail_done_offset +
            kTailDoneCounterInts * static_cast<int64_t>(sizeof(int32_t)),
        sizeof(int64_t));
    return align_i64(
        tail_signal_addrs_offset +
            kTailSignalAddrs * static_cast<int64_t>(sizeof(int64_t)),
        16);
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
