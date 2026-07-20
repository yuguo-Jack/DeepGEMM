#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>

#define K2_HIP_CHECK(expr)                                                       \
    do {                                                                        \
        const hipError_t _status = (expr);                                      \
        TORCH_CHECK(_status == hipSuccess, #expr " failed: ",                  \
                    hipGetErrorString(_status));                                \
    } while (0)

namespace {

__device__ __forceinline__ float bf16_bits_to_float(const uint16_t bits) {
    uint32_t tmp = static_cast<uint32_t>(bits) << 16;
    return __builtin_bit_cast(float, tmp);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits(const float x) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, x);
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

__device__ __forceinline__ float wave_shuffle_down_float(const float value, const int lane_delta) {
    const int32_t remote = __builtin_amdgcn_ds_bpermute(
        (__lane_id() + lane_delta) << 2,
        __builtin_bit_cast(int32_t, value));
    return __builtin_bit_cast(float, remote);
}

__device__ __forceinline__ float wave_broadcast_lane0_float(const float value) {
    const int32_t lane0 = __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int32_t, value));
    return __builtin_bit_cast(float, lane0);
}

__device__ __forceinline__ float wave_reduce_max_64(float value) {
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        value = fmaxf(value, wave_shuffle_down_float(value, offset));
    return wave_broadcast_lane0_float(value);
}

__device__ __forceinline__ float clip_e4m3fn_input(const float x) {
    return fminf(448.0f, fmaxf(-448.0f, x));
}

__device__ __forceinline__ uint8_t float_to_e4m3fn_sw(const float x) {
    constexpr int dst_exp = 4;
    constexpr int dst_mant = 3;
    constexpr int dst_bias = 7;
    constexpr int src_exp = 8;
    constexpr int src_mant = 23;
    constexpr uint32_t src_inf = 0x7f800000u;
    constexpr uint32_t src_abs_mask = 0x7fffffffu;
    constexpr uint32_t src_head_mask = 0xff800000u;
    constexpr uint32_t src_mant_mask = 0x7fffffu;
    constexpr uint32_t ifmax = 0x43e00000u; // 448.0f

    const uint32_t x_bits = __builtin_bit_cast(uint32_t, x);
    const uint32_t sign = (x_bits & src_head_mask) >> (src_exp + src_mant);
    uint32_t mantissa = x_bits & src_mant_mask;
    int exponent = static_cast<int>((x_bits & src_head_mask) >> src_mant) & 0xff;

    if ((x_bits & src_inf) == src_inf)
        return static_cast<uint8_t>((sign << (dst_exp + dst_mant)) | 0x7fu);
    if ((x_bits & src_abs_mask) > ifmax)
        return static_cast<uint8_t>((sign << (dst_exp + dst_mant)) | 0x7eu);
    if ((x_bits & src_abs_mask) == 0)
        return static_cast<uint8_t>(sign << (dst_exp + dst_mant));

    constexpr int f8_denormal_exp = 1 - dst_bias;
    int act_exponent = exponent - 127;
    int exponent_diff = 0;
    if (act_exponent <= f8_denormal_exp)
        exponent_diff = f8_denormal_exp - act_exponent;
    mantissa += 1u << src_mant;
    if (exponent_diff > dst_mant)
        return static_cast<uint8_t>(sign << (dst_exp + dst_mant));

    const int shift = src_mant - dst_mant + exponent_diff;
    const uint32_t low_mask = (1u << shift) - 1u;
    const bool midpoint = (mantissa & low_mask) == (1u << (shift - 1));
    if (exponent_diff > 0)
        mantissa >>= exponent_diff;
    const bool implicit_one = (mantissa & (1u << src_mant)) != 0;
    int f8_exponent = act_exponent + exponent_diff + dst_bias - (implicit_one ? 0 : 1);

    constexpr uint32_t drop_mask = (1u << (src_mant - dst_mant)) - 1u;
    const bool odd = (mantissa & (1u << (src_mant - dst_mant))) != 0;
    mantissa += (midpoint ? (odd ? mantissa : mantissa - 1u) : mantissa) & drop_mask;

    if (f8_exponent == 0) {
        if (mantissa & (1u << src_mant))
            f8_exponent = 1;
    } else if (mantissa & (1u << (src_mant + 1))) {
        mantissa >>= 1;
        ++f8_exponent;
    }
    mantissa >>= (src_mant - dst_mant);
    if (f8_exponent > 15)
        return static_cast<uint8_t>((sign << (dst_exp + dst_mant)) | 0x7eu);
    mantissa &= (1u << dst_mant) - 1u;
    if (f8_exponent == 15 && mantissa == 7)
        mantissa = 6;
    return static_cast<uint8_t>(
        (sign << (dst_exp + dst_mant)) |
        (static_cast<uint32_t>(f8_exponent) << dst_mant) |
        mantissa);
}

__device__ __forceinline__ uint8_t float_to_e4m3fn(const float x) {
#if defined(__gfx938__) || defined(__gfx94__) || defined(__gfx12__)
    uint32_t packed = 0;
    const float clipped = clip_e4m3fn_input(x);
    packed = __builtin_hcu_cvt_pk_fp8_f32(clipped, clipped, packed, false);
    return static_cast<uint8_t>(packed & 0xffu);
#else
    return float_to_e4m3fn_sw(x);
#endif
}

__device__ __forceinline__ uint32_t pack2_e4m3fn(const float x0, const float x1) {
#if defined(__gfx938__) || defined(__gfx94__) || defined(__gfx12__)
    uint32_t packed = 0;
    packed = __builtin_hcu_cvt_pk_fp8_f32(
        clip_e4m3fn_input(x0), clip_e4m3fn_input(x1), packed, false);
    return packed & 0xffffu;
#else
    return static_cast<uint32_t>(float_to_e4m3fn_sw(x0)) |
           (static_cast<uint32_t>(float_to_e4m3fn_sw(x1)) << 8);
#endif
}

__device__ __forceinline__ uint32_t pack4_e4m3fn(
    const float x0,
    const float x1,
    const float x2,
    const float x3) {
    const uint32_t lo = pack2_e4m3fn(x0, x1);
    const uint32_t hi = pack2_e4m3fn(x2, x3);
    return lo | (hi << 16);
}

__device__ __forceinline__ uint8_t float_to_int8_rne_bits(const float x) {
    const float rounded = nearbyintf(x);
    const int value = static_cast<int>(
        fminf(127.0f, fmaxf(-128.0f, rounded)));
    return static_cast<uint8_t>(static_cast<int8_t>(value));
}

__device__ __forceinline__ uint32_t pack4_int8_rne(
    const float x0,
    const float x1,
    const float x2,
    const float x3) {
    return static_cast<uint32_t>(float_to_int8_rne_bits(x0)) |
           (static_cast<uint32_t>(float_to_int8_rne_bits(x1)) << 8) |
           (static_cast<uint32_t>(float_to_int8_rne_bits(x2)) << 16) |
           (static_cast<uint32_t>(float_to_int8_rne_bits(x3)) << 24);
}

template <bool kInt8>
__device__ __forceinline__ uint32_t pack4_quantized(
    const float x0,
    const float x1,
    const float x2,
    const float x3) {
    if constexpr (kInt8)
        return pack4_int8_rne(x0, x1, x2, x3);
    return pack4_e4m3fn(x0, x1, x2, x3);
}

__device__ __forceinline__ int clamp_actual_m_count(
    const int count,
    const int m_per_expert) {
    if (count <= 0)
        return 0;
    return count > m_per_expert ? m_per_expert : count;
}

__device__ __forceinline__ int actual_m_max_or_scan(
    const int32_t* __restrict__ actual_m,
    const int32_t* __restrict__ actual_m_max,
    const int local_experts,
    const int m_per_expert) {
    if (actual_m_max != nullptr)
        return clamp_actual_m_count(actual_m_max[0], m_per_expert);
    int max_m = 0;
    for (int expert = 0; expert < local_experts; ++expert) {
        const int count = clamp_actual_m_count(actual_m[expert], m_per_expert);
        max_m = count > max_m ? count : max_m;
    }
    return max_m;
}

template <bool kFastMath>
__device__ __forceinline__ float swiglu_gate(const float gate) {
    if constexpr (kFastMath) {
        return gate * (1.0f / (1.0f + __expf(-gate)));
    } else {
        return gate / (1.0f + expf(-gate));
    }
}

template <int Threads, bool kFastMath, bool kInt8>
__global__ __launch_bounds__(Threads)
void swiglu_quant_channelwise_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ topk_weights,
    const int64_t* __restrict__ row_combine_ptrs,
    const int32_t* __restrict__ actual_m,
    const int32_t* __restrict__ actual_m_max,
    const int32_t* __restrict__ active_tiles,
    uint8_t* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    uint16_t* __restrict__ out_bf16,
    const int rows,
    const int hidden,
    const int m_per_expert,
    const int local_experts,
    const int active_tile_m,
    const bool has_topk_weights,
    const bool has_clamp_value,
    const bool output_bf16,
    const float clamp_value) {
    extern __shared__ float smem[];
    float* y_smem = smem;
    float* reduce_smem = smem + hidden;

    int effective_rows = rows;
    int logical_m_per_expert = m_per_expert;
    const bool has_actual_m =
        actual_m != nullptr && m_per_expert > 0 && local_experts > 0;
    const bool compact_active_rows = has_actual_m;
    if (has_actual_m) {
        const int max_m = actual_m_max_or_scan(
            actual_m, actual_m_max, local_experts, m_per_expert);
        logical_m_per_expert = max_m;
        effective_rows = max_m * local_experts;
    }
    if (!has_actual_m && active_tiles != nullptr && active_tile_m > 0) {
        const int active_rows = active_tiles[0] > 0
                                    ? active_tiles[0] * active_tile_m
                                    : 0;
        effective_rows = active_rows < effective_rows ? active_rows : effective_rows;
    }
    const int tid = static_cast<int>(threadIdx.x);
    for (int logical_row = static_cast<int>(blockIdx.x);
         logical_row < effective_rows;
         logical_row += static_cast<int>(gridDim.x)) {
        int row = logical_row;
        if (has_actual_m) {
            const int expert = logical_row / logical_m_per_expert;
            const int row_in_expert =
                logical_row - expert * logical_m_per_expert;
            const int expert_count =
                clamp_actual_m_count(actual_m[expert], m_per_expert);
            if (row_in_expert >= expert_count)
                continue;
            row = expert * m_per_expert + row_in_expert;
        }

        const int stride = hidden * 2;
        const int64_t row_base = static_cast<int64_t>(row) * stride;
        if (!compact_active_rows && !output_bf16 && row_combine_ptrs != nullptr &&
            row_combine_ptrs[row] == 0) {
            continue;
        }
        const float route_weight = has_topk_weights ? topk_weights[row] : 1.0f;
        if (!compact_active_rows && has_topk_weights && route_weight == 0.0f) {
            if (tid == 0)
                out_scale[row] = 1.0e-4f / (kInt8 ? 127.0f : 448.0f);
            const int64_t out_base = static_cast<int64_t>(row) * hidden;
            for (int col = tid * 4; col < hidden; col += Threads * 4) {
                *reinterpret_cast<uint32_t*>(out_fp8 + out_base + col) = 0u;
                if (output_bf16) {
                    uint32_t* bf16_ptr =
                        reinterpret_cast<uint32_t*>(out_bf16 + out_base + col);
                    bf16_ptr[0] = 0u;
                    bf16_ptr[1] = 0u;
                }
            }
            continue;
        }

        float local_amax = 0.0f;
        for (int col = tid * 4; col < hidden; col += Threads * 4) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float gate = bf16_bits_to_float(x[row_base + col + i]);
                float up = bf16_bits_to_float(x[row_base + hidden + col + i]);
                if (has_clamp_value) {
                    up = fminf(clamp_value, fmaxf(-clamp_value, up));
                    gate = fminf(clamp_value, gate);
                }
                const float y = swiglu_gate<kFastMath>(gate) * up * route_weight;
                y_smem[col + i] = y;
                local_amax = fmaxf(local_amax, fabsf(y));
            }
        }

        reduce_smem[tid] = local_amax;
        __syncthreads();

        for (int offset = Threads / 2; offset > 0; offset >>= 1) {
            if (tid < offset)
                reduce_smem[tid] =
                    fmaxf(reduce_smem[tid], reduce_smem[tid + offset]);
            __syncthreads();
        }

        const float clamped_amax = fmaxf(reduce_smem[0], 1.0e-4f);
        constexpr float quant_max = kInt8 ? 127.0f : 448.0f;
        const float scale = clamped_amax / quant_max;
        const float inv_scale = quant_max / clamped_amax;
        if (tid == 0)
            out_scale[row] = scale;

        const int64_t out_base = static_cast<int64_t>(row) * hidden;
        for (int col = tid * 4; col < hidden; col += Threads * 4) {
            const float y0 = y_smem[col + 0];
            const float y1 = y_smem[col + 1];
            const float y2 = y_smem[col + 2];
            const float y3 = y_smem[col + 3];
            *reinterpret_cast<uint32_t*>(out_fp8 + out_base + col) =
                pack4_quantized<kInt8>(
                    y0 * inv_scale, y1 * inv_scale, y2 * inv_scale,
                    y3 * inv_scale);
            if (output_bf16) {
                const uint32_t b01 =
                    static_cast<uint32_t>(float_to_bf16_bits(y0)) |
                    (static_cast<uint32_t>(float_to_bf16_bits(y1)) << 16);
                const uint32_t b23 =
                    static_cast<uint32_t>(float_to_bf16_bits(y2)) |
                    (static_cast<uint32_t>(float_to_bf16_bits(y3)) << 16);
                uint32_t* bf16_ptr =
                    reinterpret_cast<uint32_t*>(out_bf16 + out_base + col);
                bf16_ptr[0] = b01;
                bf16_ptr[1] = b23;
            }
        }
    }
}

template <int Threads, int VecGroups, bool kFastMath, bool kInt8>
__global__ __launch_bounds__(Threads)
void swiglu_quant_channelwise_reg_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ topk_weights,
    const int64_t* __restrict__ row_combine_ptrs,
    const int32_t* __restrict__ actual_m,
    const int32_t* __restrict__ actual_m_max,
    const int32_t* __restrict__ active_tiles,
    uint8_t* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    uint16_t* __restrict__ out_bf16,
    const int rows,
    const int m_per_expert,
    const int local_experts,
    const int active_tile_m,
    const bool has_topk_weights,
    const bool has_clamp_value,
    const bool output_bf16,
    const float clamp_value) {
    static_assert(VecGroups > 0, "VecGroups must be positive");
    constexpr int hidden = Threads * VecGroups * 4;
    extern __shared__ float reduce_smem[];

    const int tid = static_cast<int>(threadIdx.x);
    int effective_rows = rows;
    int logical_m_per_expert = m_per_expert;
    const bool has_actual_m =
        actual_m != nullptr && m_per_expert > 0 && local_experts > 0;
    const bool compact_active_rows = has_actual_m;
    if (has_actual_m) {
        const int max_m = actual_m_max_or_scan(
            actual_m, actual_m_max, local_experts, m_per_expert);
        logical_m_per_expert = max_m;
        effective_rows = max_m * local_experts;
    }
    if (!has_actual_m && active_tiles != nullptr && active_tile_m > 0) {
        const int active_rows = active_tiles[0] > 0
                                    ? active_tiles[0] * active_tile_m
                                    : 0;
        effective_rows = active_rows < effective_rows ? active_rows : effective_rows;
    }
    for (int logical_row = static_cast<int>(blockIdx.x);
         logical_row < effective_rows;
         logical_row += static_cast<int>(gridDim.x)) {
        int row = logical_row;
        if (has_actual_m) {
            const int expert = logical_row / logical_m_per_expert;
            const int row_in_expert =
                logical_row - expert * logical_m_per_expert;
            const int expert_count =
                clamp_actual_m_count(actual_m[expert], m_per_expert);
            if (row_in_expert >= expert_count)
                continue;
            row = expert * m_per_expert + row_in_expert;
        }
        const int stride = hidden * 2;
        const int64_t row_base = static_cast<int64_t>(row) * stride;
        if (!compact_active_rows && !output_bf16 && row_combine_ptrs != nullptr &&
            row_combine_ptrs[row] == 0) {
            continue;
        }
        const float route_weight = has_topk_weights ? topk_weights[row] : 1.0f;
        if (!compact_active_rows && has_topk_weights && route_weight == 0.0f) {
            if (tid == 0)
                out_scale[row] = 1.0e-4f / (kInt8 ? 127.0f : 448.0f);
            const int64_t out_base = static_cast<int64_t>(row) * hidden;
#pragma unroll
            for (int g = 0; g < VecGroups; ++g) {
                const int col = (g * Threads + tid) * 4;
                *reinterpret_cast<uint32_t*>(out_fp8 + out_base + col) = 0u;
                if (output_bf16) {
                    uint32_t* bf16_ptr = reinterpret_cast<uint32_t*>(out_bf16 + out_base + col);
                    bf16_ptr[0] = 0u;
                    bf16_ptr[1] = 0u;
                }
            }
            continue;
        }

        float y0[VecGroups];
        float y1[VecGroups];
        float y2[VecGroups];
        float y3[VecGroups];
        float local_amax = 0.0f;

#pragma unroll
        for (int g = 0; g < VecGroups; ++g) {
            const int col = (g * Threads + tid) * 4;
            float gate0 = bf16_bits_to_float(x[row_base + col + 0]);
            float gate1 = bf16_bits_to_float(x[row_base + col + 1]);
            float gate2 = bf16_bits_to_float(x[row_base + col + 2]);
            float gate3 = bf16_bits_to_float(x[row_base + col + 3]);
            float up0 = bf16_bits_to_float(x[row_base + hidden + col + 0]);
            float up1 = bf16_bits_to_float(x[row_base + hidden + col + 1]);
            float up2 = bf16_bits_to_float(x[row_base + hidden + col + 2]);
            float up3 = bf16_bits_to_float(x[row_base + hidden + col + 3]);
            if (has_clamp_value) {
                up0 = fminf(clamp_value, fmaxf(-clamp_value, up0));
                up1 = fminf(clamp_value, fmaxf(-clamp_value, up1));
                up2 = fminf(clamp_value, fmaxf(-clamp_value, up2));
                up3 = fminf(clamp_value, fmaxf(-clamp_value, up3));
                gate0 = fminf(clamp_value, gate0);
                gate1 = fminf(clamp_value, gate1);
                gate2 = fminf(clamp_value, gate2);
                gate3 = fminf(clamp_value, gate3);
            }
            y0[g] = swiglu_gate<kFastMath>(gate0) * up0 * route_weight;
            y1[g] = swiglu_gate<kFastMath>(gate1) * up1 * route_weight;
            y2[g] = swiglu_gate<kFastMath>(gate2) * up2 * route_weight;
            y3[g] = swiglu_gate<kFastMath>(gate3) * up3 * route_weight;
            local_amax = fmaxf(local_amax, fabsf(y0[g]));
            local_amax = fmaxf(local_amax, fabsf(y1[g]));
            local_amax = fmaxf(local_amax, fabsf(y2[g]));
            local_amax = fmaxf(local_amax, fabsf(y3[g]));
        }

        float row_amax = 0.0f;
        if constexpr (Threads == 64) {
            row_amax = wave_reduce_max_64(local_amax);
        } else {
            reduce_smem[tid] = local_amax;
            __syncthreads();

            for (int offset = Threads / 2; offset > 0; offset >>= 1) {
                if (tid < offset)
                    reduce_smem[tid] = fmaxf(reduce_smem[tid], reduce_smem[tid + offset]);
                __syncthreads();
            }
            row_amax = reduce_smem[0];
        }

        const float clamped_amax = fmaxf(row_amax, 1.0e-4f);
        constexpr float quant_max = kInt8 ? 127.0f : 448.0f;
        const float inv_scale = quant_max / clamped_amax;
        if (tid == 0)
            out_scale[row] = clamped_amax / quant_max;

        const int64_t out_base = static_cast<int64_t>(row) * hidden;
#pragma unroll
        for (int g = 0; g < VecGroups; ++g) {
            const int col = (g * Threads + tid) * 4;
            *reinterpret_cast<uint32_t*>(out_fp8 + out_base + col) =
                pack4_quantized<kInt8>(
                    y0[g] * inv_scale, y1[g] * inv_scale,
                    y2[g] * inv_scale, y3[g] * inv_scale);
            if (output_bf16) {
                const uint32_t b01 =
                    static_cast<uint32_t>(float_to_bf16_bits(y0[g])) |
                    (static_cast<uint32_t>(float_to_bf16_bits(y1[g])) << 16);
                const uint32_t b23 =
                    static_cast<uint32_t>(float_to_bf16_bits(y2[g])) |
                    (static_cast<uint32_t>(float_to_bf16_bits(y3[g])) << 16);
                uint32_t* bf16_ptr = reinterpret_cast<uint32_t*>(out_bf16 + out_base + col);
                bf16_ptr[0] = b01;
                bf16_ptr[1] = b23;
            }
        }
        if constexpr (Threads != 64) {
            // Only needed between grid-stride row iterations that reuse shared
            // reduction storage. The default one-row-per-block path avoids an
            // extra barrier and stays close to the original launch behavior.
            if (row + static_cast<int>(gridDim.x) >= rows)
                continue;
            __syncthreads();
        }
    }
}

template <int Threads, bool kFastMath, bool kInt8>
void launch_swiglu_quant_channelwise(
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    const int64_t* row_combine_ptrs,
    const int32_t* actual_m,
    const int32_t* actual_m_max,
    const int32_t* active_tiles,
    torch::Tensor& out_fp8,
    torch::Tensor& out_scale,
    torch::Tensor& out_bf16,
    const bool output_bf16,
    const bool has_clamp_value,
    const double clamp_value,
    const int64_t max_row_blocks,
    const int64_t m_per_expert,
    const int64_t local_experts,
    const int64_t active_tile_m) {
    const int rows = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1) / 2);
    if (rows == 0)
        return;
    const int launch_blocks =
        max_row_blocks > 0
            ? std::max<int>(1, std::min<int64_t>(rows, max_row_blocks))
            : rows;
    const bool has_topk_weights = topk_weights.numel() > 0;
    const size_t shared_bytes_for_reg =
        Threads == 64 ? 0 : static_cast<size_t>(Threads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (hidden == 2048) {
        constexpr int vec_groups = 2048 / (Threads * 4);
        hipLaunchKernelGGL(
            (swiglu_quant_channelwise_reg_kernel<
                Threads, vec_groups, kFastMath, kInt8>),
            dim3(launch_blocks),
            dim3(Threads),
            shared_bytes_for_reg,
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
            has_topk_weights ? topk_weights.data_ptr<float>() : nullptr,
            row_combine_ptrs,
            actual_m,
            actual_m_max,
            active_tiles,
            reinterpret_cast<uint8_t*>(out_fp8.data_ptr()),
            out_scale.data_ptr<float>(),
            output_bf16 ? reinterpret_cast<uint16_t*>(out_bf16.data_ptr<at::BFloat16>()) : nullptr,
            rows,
            static_cast<int>(m_per_expert),
            static_cast<int>(local_experts),
            static_cast<int>(active_tile_m),
            has_topk_weights,
            has_clamp_value,
            output_bf16,
            static_cast<float>(clamp_value));
    } else if (hidden == 4096) {
        constexpr int vec_groups = 4096 / (Threads * 4);
        hipLaunchKernelGGL(
            (swiglu_quant_channelwise_reg_kernel<
                Threads, vec_groups, kFastMath, kInt8>),
            dim3(launch_blocks),
            dim3(Threads),
            shared_bytes_for_reg,
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
            has_topk_weights ? topk_weights.data_ptr<float>() : nullptr,
            row_combine_ptrs,
            actual_m,
            actual_m_max,
            active_tiles,
            reinterpret_cast<uint8_t*>(out_fp8.data_ptr()),
            out_scale.data_ptr<float>(),
            output_bf16 ? reinterpret_cast<uint16_t*>(out_bf16.data_ptr<at::BFloat16>()) : nullptr,
            rows,
            static_cast<int>(m_per_expert),
            static_cast<int>(local_experts),
            static_cast<int>(active_tile_m),
            has_topk_weights,
            has_clamp_value,
            output_bf16,
            static_cast<float>(clamp_value));
    } else {
        const size_t shared_bytes =
            static_cast<size_t>(hidden + Threads) * sizeof(float);
        hipLaunchKernelGGL(
            (swiglu_quant_channelwise_kernel<Threads, kFastMath, kInt8>),
            dim3(launch_blocks),
            dim3(Threads),
            shared_bytes,
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
            has_topk_weights ? topk_weights.data_ptr<float>() : nullptr,
            row_combine_ptrs,
            actual_m,
            actual_m_max,
            active_tiles,
            reinterpret_cast<uint8_t*>(out_fp8.data_ptr()),
            out_scale.data_ptr<float>(),
            output_bf16 ? reinterpret_cast<uint16_t*>(out_bf16.data_ptr<at::BFloat16>()) : nullptr,
            rows,
            hidden,
            static_cast<int>(m_per_expert),
            static_cast<int>(local_experts),
            static_cast<int>(active_tile_m),
            has_topk_weights,
            has_clamp_value,
            output_bf16,
            static_cast<float>(clamp_value));
    }
    K2_HIP_CHECK(hipGetLastError());
}

template <bool kFastMath, bool kInt8>
void launch_swiglu_quant_channelwise_auto(
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    const int64_t* row_combine_ptrs,
    const int32_t* actual_m,
    const int32_t* actual_m_max,
    const int32_t* active_tiles,
    torch::Tensor& out_fp8,
    torch::Tensor& out_scale,
    torch::Tensor& out_bf16,
    const bool output_bf16,
    const bool has_clamp_value,
    const double clamp_value,
    const int64_t max_row_blocks,
    const int64_t m_per_expert,
    const int64_t local_experts,
    const int64_t active_tile_m) {
    const int hidden = static_cast<int>(x.size(1) / 2);
    if (hidden <= 2048 && !output_bf16) {
        launch_swiglu_quant_channelwise<64, kFastMath, kInt8>(
            x, topk_weights, row_combine_ptrs, actual_m, actual_m_max,
            active_tiles,
            out_fp8, out_scale, out_bf16,
            output_bf16, has_clamp_value, clamp_value, max_row_blocks,
            m_per_expert, local_experts, active_tile_m);
    } else if (hidden <= 2048) {
        launch_swiglu_quant_channelwise<128, kFastMath, kInt8>(
            x, topk_weights, row_combine_ptrs, actual_m, actual_m_max,
            active_tiles,
            out_fp8, out_scale, out_bf16,
            output_bf16, has_clamp_value, clamp_value, max_row_blocks,
            m_per_expert, local_experts, active_tile_m);
    } else if (hidden == 4096) {
        launch_swiglu_quant_channelwise<128, kFastMath, kInt8>(
            x, topk_weights, row_combine_ptrs, actual_m, actual_m_max,
            active_tiles,
            out_fp8, out_scale, out_bf16,
            output_bf16, has_clamp_value, clamp_value, max_row_blocks,
            m_per_expert, local_experts, active_tile_m);
    } else {
        launch_swiglu_quant_channelwise<256, kFastMath, kInt8>(
            x, topk_weights, row_combine_ptrs, actual_m, actual_m_max,
            active_tiles,
            out_fp8, out_scale, out_bf16,
            output_bf16, has_clamp_value, clamp_value, max_row_blocks,
            m_per_expert, local_experts, active_tile_m);
    }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA/HIP");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

void swiglu_quant_channelwise_out_impl(
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    const torch::Tensor& out_fp8,
    const torch::Tensor& out_scale,
    const torch::Tensor& out_bf16,
    const bool output_bf16,
    const bool has_clamp_value,
    const double clamp_value,
    const std::optional<torch::Tensor>& row_combine_ptrs,
    const int64_t max_row_blocks,
    const std::optional<torch::Tensor>& actual_m,
    const int64_t m_per_expert,
    const std::optional<torch::Tensor>& active_tiles,
    const int64_t active_tile_m,
    const bool fast_math,
    const bool int8_output) {
    check_cuda_contiguous(x, "x");
    check_cuda_contiguous(out_fp8, "out_fp8");
    check_cuda_contiguous(out_scale, "out_scale");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(
        out_fp8.scalar_type() ==
            (int8_output ? torch::kInt8 : torch::kFloat8_e4m3fn),
        int8_output ? "out_int8 must be INT8" : "out_fp8 must be FP8 E4M3");
    TORCH_CHECK(out_scale.scalar_type() == torch::kFloat32,
                "out_scale must be FP32");
    TORCH_CHECK(x.dim() == 2 && x.size(1) % 2 == 0,
                "x must be [rows, 2 * hidden]");
    const int64_t rows = x.size(0);
    const int64_t hidden = x.size(1) / 2;
    TORCH_CHECK(hidden > 0 && hidden <= 4096,
                "K2 fused currently supports hidden in (0, 4096]");
    TORCH_CHECK(hidden % 4 == 0,
                "K2 fused vectorized path requires hidden to be divisible by 4");
    TORCH_CHECK(out_fp8.dim() == 2 && out_fp8.size(0) == rows &&
                    out_fp8.size(1) == hidden,
                "out_fp8 shape must be [rows, hidden]");
    TORCH_CHECK(out_scale.numel() >= rows,
                "out_scale must have at least one scale per row");
    if (topk_weights.numel() > 0) {
        check_cuda_contiguous(topk_weights, "topk_weights");
        TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32,
                    "topk_weights must be FP32");
        TORCH_CHECK(topk_weights.numel() >= rows,
                    "topk_weights must cover every row");
    }
    const int64_t* row_combine_ptrs_data = nullptr;
    if (row_combine_ptrs.has_value()) {
        const auto& ptrs = row_combine_ptrs.value();
        check_cuda_contiguous(ptrs, "row_combine_ptrs");
        TORCH_CHECK(ptrs.scalar_type() == torch::kInt64,
                    "row_combine_ptrs must be int64");
        TORCH_CHECK(ptrs.numel() >= rows,
                    "row_combine_ptrs must cover every row");
        row_combine_ptrs_data = ptrs.data_ptr<int64_t>();
    }
    const int32_t* actual_m_data = nullptr;
    const int32_t* actual_m_max_data = nullptr;
    int64_t local_experts = 0;
    if (actual_m.has_value()) {
        const auto& actual = actual_m.value();
        check_cuda_contiguous(actual, "actual_m");
        TORCH_CHECK(actual.scalar_type() == torch::kInt,
                    "actual_m must be int32");
        TORCH_CHECK(actual.numel() > 0, "actual_m must be non-empty");
        TORCH_CHECK(m_per_expert > 0 && rows % m_per_expert == 0,
                    "m_per_expert must evenly divide rows when actual_m is set");
        local_experts = rows / m_per_expert;
        TORCH_CHECK(actual.numel() >= local_experts,
                    "actual_m must cover every local expert");
        actual_m_data = actual.data_ptr<int32_t>();
        if (actual.numel() > local_experts)
            actual_m_max_data = actual_m_data + local_experts;
    }
    const int32_t* active_tiles_data = nullptr;
    if (active_tiles.has_value()) {
        const auto& tiles = active_tiles.value();
        check_cuda_contiguous(tiles, "active_tiles");
        TORCH_CHECK(tiles.scalar_type() == torch::kInt,
                    "active_tiles must be int32");
        TORCH_CHECK(tiles.numel() >= 1,
                    "active_tiles must contain at least one int32");
        TORCH_CHECK(active_tile_m > 0,
                    "active_tile_m must be positive when active_tiles is set");
        active_tiles_data = tiles.data_ptr<int32_t>();
    }
    if (output_bf16) {
        check_cuda_contiguous(out_bf16, "out_bf16");
        TORCH_CHECK(out_bf16.scalar_type() == torch::kBFloat16,
                    "out_bf16 must be BF16");
        TORCH_CHECK(out_bf16.dim() == 2 && out_bf16.size(0) == rows &&
                        out_bf16.size(1) == hidden,
                    "out_bf16 shape must be [rows, hidden]");
    }

    auto out_fp8_view = out_fp8;
    auto out_scale_view = out_scale.view({-1});
    auto out_bf16_view = out_bf16;
    if (fast_math) {
        if (int8_output) {
            launch_swiglu_quant_channelwise_auto<true, true>(
                x, topk_weights, row_combine_ptrs_data, actual_m_data,
                actual_m_max_data, active_tiles_data,
                out_fp8_view, out_scale_view, out_bf16_view,
                output_bf16, has_clamp_value, clamp_value, max_row_blocks,
                m_per_expert, local_experts, active_tile_m);
        } else {
            launch_swiglu_quant_channelwise_auto<true, false>(
                x, topk_weights, row_combine_ptrs_data, actual_m_data,
                actual_m_max_data, active_tiles_data,
                out_fp8_view, out_scale_view, out_bf16_view,
                output_bf16, has_clamp_value, clamp_value, max_row_blocks,
                m_per_expert, local_experts, active_tile_m);
        }
    } else {
        if (int8_output) {
            launch_swiglu_quant_channelwise_auto<false, true>(
                x, topk_weights, row_combine_ptrs_data, actual_m_data,
                actual_m_max_data, active_tiles_data,
                out_fp8_view, out_scale_view, out_bf16_view,
                output_bf16, has_clamp_value, clamp_value, max_row_blocks,
                m_per_expert, local_experts, active_tile_m);
        } else {
            launch_swiglu_quant_channelwise_auto<false, false>(
                x, topk_weights, row_combine_ptrs_data, actual_m_data,
                actual_m_max_data, active_tiles_data,
                out_fp8_view, out_scale_view, out_bf16_view,
                output_bf16, has_clamp_value, clamp_value, max_row_blocks,
                m_per_expert, local_experts, active_tile_m);
        }
    }
}

#define K2_SWIGLU_QUANT_ARGS                                                   \
    const torch::Tensor& x, const torch::Tensor& topk_weights,                 \
    const torch::Tensor& out_quant, const torch::Tensor& out_scale,            \
    const torch::Tensor& out_bf16, const bool output_bf16,                     \
    const bool has_clamp_value, const double clamp_value,                      \
    const std::optional<torch::Tensor>& row_combine_ptrs,                      \
    const int64_t max_row_blocks, const std::optional<torch::Tensor>& actual_m,\
    const int64_t m_per_expert, const std::optional<torch::Tensor>& active_tiles,\
    const int64_t active_tile_m, const bool fast_math

void swiglu_quant_channelwise_out(K2_SWIGLU_QUANT_ARGS) {
    swiglu_quant_channelwise_out_impl(
        x, topk_weights, out_quant, out_scale, out_bf16, output_bf16,
        has_clamp_value, clamp_value, row_combine_ptrs, max_row_blocks,
        actual_m, m_per_expert, active_tiles, active_tile_m, fast_math, false);
}

void swiglu_quant_int8_channelwise_out(K2_SWIGLU_QUANT_ARGS) {
    swiglu_quant_channelwise_out_impl(
        x, topk_weights, out_quant, out_scale, out_bf16, output_bf16,
        has_clamp_value, clamp_value, row_combine_ptrs, max_row_blocks,
        actual_m, m_per_expert, active_tiles, active_tile_m, fast_math, true);
}

#undef K2_SWIGLU_QUANT_ARGS

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_quant_channelwise_out", &swiglu_quant_channelwise_out,
          pybind11::arg("x"),
          pybind11::arg("topk_weights"),
          pybind11::arg("out_fp8"),
          pybind11::arg("out_scale"),
          pybind11::arg("out_bf16"),
          pybind11::arg("output_bf16") = false,
          pybind11::arg("has_clamp_value") = true,
          pybind11::arg("clamp_value") = 10.0,
          pybind11::arg("row_combine_ptrs") = std::nullopt,
          pybind11::arg("max_row_blocks") = -1,
          pybind11::arg("actual_m") = std::nullopt,
          pybind11::arg("m_per_expert") = 0,
          pybind11::arg("active_tiles") = std::nullopt,
          pybind11::arg("active_tile_m") = 0,
          pybind11::arg("fast_math") = true);
    m.def("swiglu_quant_int8_channelwise_out",
          &swiglu_quant_int8_channelwise_out,
          pybind11::arg("x"),
          pybind11::arg("topk_weights"),
          pybind11::arg("out_int8"),
          pybind11::arg("out_scale"),
          pybind11::arg("out_bf16"),
          pybind11::arg("output_bf16") = false,
          pybind11::arg("has_clamp_value") = true,
          pybind11::arg("clamp_value") = 10.0,
          pybind11::arg("row_combine_ptrs") = std::nullopt,
          pybind11::arg("max_row_blocks") = -1,
          pybind11::arg("actual_m") = std::nullopt,
          pybind11::arg("m_per_expert") = 0,
          pybind11::arg("active_tiles") = std::nullopt,
          pybind11::arg("active_tile_m") = 0,
          pybind11::arg("fast_math") = true);
}
