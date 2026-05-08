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
    if (!fast_math || hidden % 32 != 0 || intermediate_hidden % 32 != 0) {
        throw std::runtime_error(
            "DCU MegaMoE W8A8 channelwise persistent fused path requires "
            "fast_math and hidden/intermediate_hidden divisible by 32");
    }

    // The x-pool path has a wider dispatch-pull copy before GEMM. On gfx938,
    // 256 threads and up to 128 persistent blocks gave the best EP8/512 balance
    // between pull parallelism, wave pressure, and grid-barrier overhead.
    constexpr int threads = 256;
    const int blocks = std::min(std::max(num_experts_per_rank * 4, 1), 128);
    const int route_tile_log2_m = choose_dcu_route_tile_log2_m(
        num_ranks, num_tokens, num_topk, num_experts_per_rank);
    const int route_tile_m = dcu_route_tile_m_from_log2(route_tile_log2_m);

    size_t smem_bytes = 0;
    uint8_t* route_scratch = nullptr;
    int64_t route_scratch_tiles = 0;
    const int64_t total_route_tasks =
        static_cast<int64_t>(num_ranks) * num_tokens * num_topk;
    route_scratch_tiles =
        num_experts_per_rank + (total_route_tasks + route_tile_m - 1) / route_tile_m;
    const auto layout = dcu_route_tile_scratch_layout(
        route_scratch_tiles, route_tile_m, hidden, intermediate_hidden);
    const size_t scratch_bytes = static_cast<size_t>(layout.bytes);

    static uint8_t* persistent_route_scratch = nullptr;
    static size_t persistent_route_scratch_bytes = 0;
    if (scratch_bytes > persistent_route_scratch_bytes) {
        if (persistent_route_scratch != nullptr)
            DG_HIP_CHECK(hipFree(persistent_route_scratch));
        DG_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&persistent_route_scratch), scratch_bytes));
        persistent_route_scratch_bytes = scratch_bytes;
    }
    route_scratch = persistent_route_scratch;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    uint8_t** device_sym_buffer_ptrs = get_cached_device_sym_buffer_ptrs(sym_buffer_ptrs, num_ranks, stream);
    int** device_signal_ptrs = get_cached_device_signal_ptrs(signal_ptrs, num_ranks, stream);
#define LAUNCH_MEGAMOE_DCU_ROUTE_TILE(LOG2_M)                                      \
    hipLaunchKernelGGL(                                                            \
        HIP_KERNEL_NAME(mega_moe_multirank_persistent_w8a8_channelwise_kernel<LOG2_M>), \
        dim3(blocks), dim3(threads), smem_bytes, stream,                           \
        static_cast<uint16_t*>(y),                                                 \
        device_sym_buffer_ptrs, device_signal_ptrs,                                \
        route_scratch, route_scratch_tiles,                                        \
        static_cast<const uint8_t*>(l1_weights), l1_weights_sf,                    \
        static_cast<const uint8_t*>(l2_weights), l2_weights_sf,                    \
        cumulative_local_expert_recv_stats,                                        \
        rank_idx, num_ranks, num_max_tokens_per_rank,                              \
        num_experts_per_rank, num_tokens, num_topk,                                \
        hidden, intermediate_hidden, activation_clamp)

#define LAUNCH_MEGAMOE_DCU_V4_EP8()                                                \
    hipLaunchKernelGGL(                                                            \
        HIP_KERNEL_NAME(mega_moe_multirank_persistent_w8a8_channelwise_kernel<     \
            kDcuRouteTileMMinLog2, 8, 6, 4096, 2048, 32>),                         \
        dim3(blocks), dim3(threads), smem_bytes, stream,                           \
        static_cast<uint16_t*>(y),                                                 \
        device_sym_buffer_ptrs, device_signal_ptrs,                                \
        route_scratch, route_scratch_tiles,                                        \
        static_cast<const uint8_t*>(l1_weights), l1_weights_sf,                    \
        static_cast<const uint8_t*>(l2_weights), l2_weights_sf,                    \
        cumulative_local_expert_recv_stats,                                        \
        rank_idx, num_ranks, num_max_tokens_per_rank,                              \
        num_experts_per_rank, num_tokens, num_topk,                                \
        hidden, intermediate_hidden, activation_clamp)

    if (num_ranks == 8 && num_topk == 6 && hidden == 4096 && intermediate_hidden == 2048 &&
        num_experts_per_rank == 32) {
        LAUNCH_MEGAMOE_DCU_V4_EP8();
    } else if (route_tile_log2_m == kDcuRouteTileMMinLog2) {
        LAUNCH_MEGAMOE_DCU_ROUTE_TILE(kDcuRouteTileMMinLog2);
    } else if (route_tile_log2_m == kDcuRouteTileMMidLog2) {
        LAUNCH_MEGAMOE_DCU_ROUTE_TILE(kDcuRouteTileMMidLog2);
    } else {
        LAUNCH_MEGAMOE_DCU_ROUTE_TILE(kDcuRouteTileMLargeLog2);
    }
#undef LAUNCH_MEGAMOE_DCU_V4_EP8
#undef LAUNCH_MEGAMOE_DCU_ROUTE_TILE
    DG_HIP_CHECK(hipGetLastError());
}


} // namespace deep_gemm::mega
