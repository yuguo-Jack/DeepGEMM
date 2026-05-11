#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <deep_gemm/impls/mega_moe_dcu.cuh>



namespace deep_gemm::mega {

static uint8_t** get_cached_device_sym_buffer_ptrs(
    const int64_t* sym_buffer_ptrs,
    const int num_ranks,
    hipStream_t stream) {
    static uint8_t** device_ptrs = nullptr;
    static int capacity = 0;
    static std::vector<uint8_t*> host_ptrs;

    if (num_ranks > capacity) {
        if (device_ptrs != nullptr)
            DG_HIP_CHECK(hipFree(device_ptrs));
        DG_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_ptrs), sizeof(uint8_t*) * num_ranks));
        capacity = num_ranks;
    }
    host_ptrs.resize(num_ranks);
    for (int i = 0; i < num_ranks; ++i)
        host_ptrs[i] = reinterpret_cast<uint8_t*>(sym_buffer_ptrs[i]);
    DG_HIP_CHECK(hipMemcpyAsync(
        device_ptrs, host_ptrs.data(), sizeof(uint8_t*) * num_ranks,
        hipMemcpyHostToDevice, stream));
    return device_ptrs;
}

static int** get_cached_device_signal_ptrs(
    const int64_t* signal_ptrs,
    const int num_ranks,
    hipStream_t stream) {
    static int** device_ptrs = nullptr;
    static int capacity = 0;
    static std::vector<int*> host_ptrs;

    if (num_ranks > capacity) {
        if (device_ptrs != nullptr)
            DG_HIP_CHECK(hipFree(device_ptrs));
        DG_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_ptrs), sizeof(int*) * num_ranks));
        capacity = num_ranks;
    }
    host_ptrs.resize(num_ranks);
    for (int i = 0; i < num_ranks; ++i)
        host_ptrs[i] = reinterpret_cast<int*>(signal_ptrs[i]);
    DG_HIP_CHECK(hipMemcpyAsync(
        device_ptrs, host_ptrs.data(), sizeof(int*) * num_ranks,
        hipMemcpyHostToDevice, stream));
    return device_ptrs;
}

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
    const int64_t* signal_ptrs,
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
    const auto layout = dcu_route_tile_scratch_layout(
        route_scratch_tiles, route_tile_m, Shape::kHidden, Shape::kIntermediate);
    const size_t scratch_bytes = static_cast<size_t>(layout.bytes);

    static uint8_t* persistent_route_scratch = nullptr;
    static size_t persistent_route_scratch_bytes = 0;
    if (scratch_bytes > persistent_route_scratch_bytes) {
        if (persistent_route_scratch != nullptr)
            DG_HIP_CHECK(hipFree(persistent_route_scratch));
        DG_HIP_CHECK(hipMalloc(
            reinterpret_cast<void**>(&persistent_route_scratch), scratch_bytes));
        persistent_route_scratch_bytes = scratch_bytes;
    }

    uint8_t** device_sym_buffer_ptrs =
        get_cached_device_sym_buffer_ptrs(sym_buffer_ptrs, Shape::kNumRanks, stream);
    int** device_signal_ptrs =
        get_cached_device_signal_ptrs(signal_ptrs, Shape::kNumRanks, stream);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mega_moe_multirank_persistent_w8a8_channelwise_kernel<
            KernelConfig>),
        dim3(blocks), dim3(threads), 0, stream,
        static_cast<uint16_t*>(y),
        device_sym_buffer_ptrs, device_signal_ptrs,
        persistent_route_scratch, route_scratch_tiles,
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
    const int64_t* signal_ptrs,
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
            sym_buffer_ptrs, signal_ptrs,
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
