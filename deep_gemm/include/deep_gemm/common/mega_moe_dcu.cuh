#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

#include <deep_gemm/comm/mega_moe_dcu.cuh>
#include <deep_gemm/mma/fp8_mmac_dcu.cuh>

namespace deep_gemm::mega {

#define DG_HIP_CHECK(expr) do { \
    const hipError_t status = (expr); \
    if (status != hipSuccess) { \
        throw std::runtime_error(std::string(#expr) + " failed: " + hipGetErrorString(status)); \
    } \
} while (0)


__device__ static inline uint16_t float_to_bf16_bits(const float x) {
    uint32_t bits = __float_as_uint(x);
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

__device__ static inline float bf16_bits_to_float(const uint16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
}

__device__ static inline dcu::int32x2_t pack8_fp8_weight_marlin(const uint8_t* weight,
                                                                const int expert_idx,
                                                                const int row_idx,
                                                                const int k_idx,
                                                                const int rows,
                                                                const int k) {
    return dcu::pack8_fp8(weight + marlin_nt_kpack2_offset(expert_idx, row_idx, k_idx, rows, k));
}

__device__ static inline float load_weight_scale_channelwise(const float* sf,
                                                            const int expert_idx,
                                                            const int row_idx,
                                                            const int rows) {
    const int64_t row = static_cast<int64_t>(expert_idx) * rows + row_idx;
    return sf[row];
}

__device__ static inline float vec4_get(const dcu::float32x4_t value, const int slot) {
    return slot == 0 ? value.x : (slot == 1 ? value.y : (slot == 2 ? value.z : value.w));
}

__device__ static inline float fast_silu(const float x) {
#if defined(__gfx938__)
    constexpr float kLog2E = 1.4426950408889634f;
    const float exp_val = __builtin_amdgcn_exp2f(-x * kLog2E);
    const float sigmoid = __builtin_amdgcn_rcpf(1.0f + exp_val);
    return x * sigmoid;
#else
    return x / (1.0f + expf(-x));
#endif
}

} // namespace deep_gemm::mega
