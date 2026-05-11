#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <deep_gemm/impls/mega_moe_dcu.cuh>

namespace deep_gemm::mega {

template <typename KernelConfig>
static bool dcu_mega_moe_shape_matches(
    const int num_ranks,
    const int num_experts_per_rank,
    const int num_topk,
    const int hidden,
    const int intermediate_hidden) {
    using Shape = typename KernelConfig::Shape;
    return num_ranks == Shape::kNumRanks &&
           num_experts_per_rank == Shape::kExpertsPerRank &&
           num_ranks * num_experts_per_rank == Shape::kNumExperts &&
           num_topk == Shape::kTopK &&
           hidden == Shape::kHidden &&
           intermediate_hidden == Shape::kIntermediate;
}

template <typename KernelConfig>
static void launch_mega_moe_dcu_template(
    void* y,
    const void* l1_weights, const float* l1_weights_sf,
    const void* l2_weights, const float* l2_weights_sf,
    int* cumulative_local_expert_recv_stats,
    const int64_t* sym_buffer_ptrs,
    void* route_scratch,
    int rank_idx,
    int num_max_tokens_per_rank,
    int num_tokens,
    float activation_clamp,
    hipStream_t stream) {
    using Shape = typename KernelConfig::Shape;

    constexpr int threads = KernelConfig::kThreads;
    constexpr int blocks = KernelConfig::kBlocks;
    constexpr int route_tile_m = 1 << KernelConfig::kRouteTileLog2M;

    const int64_t total_route_tasks =
        static_cast<int64_t>(Shape::kNumRanks) * num_tokens * Shape::kTopK;
    const int64_t route_scratch_tiles =
        Shape::kExpertsPerRank + (total_route_tasks + route_tile_m - 1) / route_tile_m;
    auto* local_sym_buffer = reinterpret_cast<uint8_t*>(sym_buffer_ptrs[rank_idx]);
    uint8_t** device_sym_buffer_ptrs = dcu_peer_sym_buffer_ptrs(local_sym_buffer);
    int** device_signal_ptrs = dcu_peer_signal_ptrs(local_sym_buffer, Shape::kNumRanks);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mega_moe_multirank_persistent_w8a8_channelwise_kernel<
            KernelConfig>),
        dim3(blocks), dim3(threads), 0, stream,
        static_cast<uint16_t*>(y),
        device_sym_buffer_ptrs, device_signal_ptrs,
        static_cast<uint8_t*>(route_scratch), route_scratch_tiles,
        static_cast<const uint8_t*>(l1_weights), l1_weights_sf,
        static_cast<const uint8_t*>(l2_weights), l2_weights_sf,
        cumulative_local_expert_recv_stats,
        rank_idx, num_max_tokens_per_rank, num_tokens, activation_clamp);
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
    bool fast_math) {
    if (!fast_math) {
        throw std::runtime_error(
            "DCU MegaMoE W8A8 channelwise persistent fused path requires "
            "fast_math");
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (dcu_mega_moe_shape_matches<DcuMegaMoeEp8Config>(
            num_ranks, num_experts_per_rank, num_topk, hidden, intermediate_hidden)) {
        launch_mega_moe_dcu_template<DcuMegaMoeEp8Config>(
            y,
            l1_weights, l1_weights_sf,
            l2_weights, l2_weights_sf,
            cumulative_local_expert_recv_stats,
            sym_buffer_ptrs, route_scratch,
            rank_idx, num_max_tokens_per_rank, num_tokens,
            activation_clamp, stream);
    } else {
        throw std::runtime_error(
            "Unsupported DCU MegaMoE W8A8 fused shape. Registered templates: "
            "EP8, num_experts=256, topk=6, hidden=4096, intermediate=2048.");
    }
    DG_HIP_CHECK(hipGetLastError());
}


} // namespace deep_gemm::mega
