#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

namespace deep_gemm::mega {

#define DG_HIP_CHECK(expr) do { \
    const hipError_t status = (expr); \
    if (status != hipSuccess) { \
        throw std::runtime_error(std::string(#expr) + " failed: " + hipGetErrorString(status)); \
    } \
} while (0)

__device__ static inline uint16_t float_to_bf16_bits(const float x) {
#if defined(__gfx938__)
    return static_cast<uint16_t>(__builtin_hcu_cvt_bf16_f32(x, false, false));
#else
    uint32_t bits = __float_as_uint(x);
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
#endif
}

__device__ static inline float bf16_bits_to_float(const uint16_t x) {
#if defined(__gfx938__)
    return __builtin_hcu_cvt_f32_bf16(x, false, 0, false);
#else
    return __uint_as_float(static_cast<uint32_t>(x) << 16);
#endif
}

__device__ static inline float wave_shuffle_down_float(const float value, const int lane_delta) {
    const int32_t remote = __builtin_amdgcn_ds_bpermute(
        (__lane_id() + lane_delta) << 2,
        __builtin_bit_cast(int32_t, value));
    return __builtin_bit_cast(float, remote);
}

__device__ static inline float wave_broadcast_lane0_float(const float value) {
    const int32_t lane0 = __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int32_t, value));
    return __builtin_bit_cast(float, lane0);
}

__device__ static inline float wave_reduce_max_64(float value) {
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        value = fmaxf(value, wave_shuffle_down_float(value, offset));
    return wave_broadcast_lane0_float(value);
}

__device__ static inline float clip_fp8_e4m3fn(const float x) {
    return fminf(448.0f, fmaxf(-448.0f, x));
}

__device__ static inline uint32_t pack2_fp8_e4m3fn(const float x0, const float x1) {
#if defined(__gfx938__) || defined(__gfx94__) || defined(__gfx12__)
    uint32_t packed = 0;
    packed = __builtin_hcu_cvt_pk_fp8_f32(
        clip_fp8_e4m3fn(x0), clip_fp8_e4m3fn(x1), packed, false);
    return packed & 0xffffu;
#else
    return 0u;
#endif
}

__device__ static inline uint32_t pack4_fp8_e4m3fn(
    const float x0,
    const float x1,
    const float x2,
    const float x3) {
#if defined(__gfx938__) || defined(__gfx94__) || defined(__gfx12__)
    uint32_t packed = 0;
    packed = __builtin_hcu_cvt_pk_fp8_f32(
        clip_fp8_e4m3fn(x0), clip_fp8_e4m3fn(x1), packed, false);
    packed = __builtin_hcu_cvt_pk_fp8_f32(
        clip_fp8_e4m3fn(x2), clip_fp8_e4m3fn(x3), packed, true);
    return packed;
#else
    const uint32_t lo = pack2_fp8_e4m3fn(x0, x1);
    const uint32_t hi = pack2_fp8_e4m3fn(x2, x3);
    return lo | (hi << 16);
#endif
}

__device__ static inline uint8_t float_to_int8_rne_bits(const float x) {
    const float rounded = nearbyintf(x);
    const int value = static_cast<int>(
        fminf(127.0f, fmaxf(-128.0f, rounded)));
    return static_cast<uint8_t>(static_cast<int8_t>(value));
}

__device__ static inline uint32_t pack4_int8_rne(
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
__device__ static inline uint32_t pack4_channelwise_quant(
    const float x0,
    const float x1,
    const float x2,
    const float x3) {
    if constexpr (kInt8)
        return pack4_int8_rne(x0, x1, x2, x3);
    return pack4_fp8_e4m3fn(x0, x1, x2, x3);
}

template <bool TopkIdxI64, bool TopkWeightsBf16>
__device__ static inline void stage_topk_route(const void* __restrict__ topk_idx,
                                               const void* __restrict__ topk_weights,
                                               int64_t* __restrict__ out_topk_idx,
                                               float* __restrict__ out_topk_weights,
                                               const int64_t route_offset) {
    if constexpr (TopkIdxI64) {
        out_topk_idx[route_offset] = static_cast<const int64_t*>(topk_idx)[route_offset];
    } else {
        out_topk_idx[route_offset] =
            static_cast<int64_t>(static_cast<const int*>(topk_idx)[route_offset]);
    }

    if constexpr (TopkWeightsBf16) {
        out_topk_weights[route_offset] =
            bf16_bits_to_float(static_cast<const uint16_t*>(topk_weights)[route_offset]);
    } else {
        out_topk_weights[route_offset] = static_cast<const float*>(topk_weights)[route_offset];
    }
}

template <int Threads, bool TopkIdxI64, bool TopkWeightsBf16, bool kInt8>
__global__ __launch_bounds__(Threads)
void mega_moe_pre_dispatch_fp8_channelwise_kernel(const uint16_t* __restrict__ x_bf16,
                                                  const void* __restrict__ topk_idx,
                                                  const void* __restrict__ topk_weights,
                                                  uint8_t* __restrict__ out_fp8,
                                                  float* __restrict__ out_scale,
                                                  int64_t* __restrict__ out_topk_idx,
                                                  float* __restrict__ out_topk_weights,
                                                  const int rows,
                                                  const int hidden,
                                                  const int topk) {
    __shared__ float wave_maxes[Threads / 64];
    __shared__ float row_scale;

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;

    const int tid = static_cast<int>(threadIdx.x);
    const int64_t x_row_offset = static_cast<int64_t>(row) * hidden;
    const int64_t topk_row_offset = static_cast<int64_t>(row) * topk;

    if (tid < topk) {
        const int64_t route_offset = topk_row_offset + tid;
        stage_topk_route<TopkIdxI64, TopkWeightsBf16>(
            topk_idx, topk_weights, out_topk_idx, out_topk_weights, route_offset);
    }

    float local_max = 0.0f;
    for (int col = tid * 8; col < hidden; col += Threads * 8) {
        const uint64_t packed =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col);
        const uint64_t packed_hi =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 4);
        const float x0 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed)));
        const float x1 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed >> 16)));
        const float x2 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed >> 32)));
        const float x3 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed >> 48)));
        const float x4 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed_hi)));
        const float x5 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 16)));
        const float x6 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 32)));
        const float x7 = fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 48)));
        local_max = fmaxf(local_max,
                          fmaxf(fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)),
                                fmaxf(fmaxf(x4, x5), fmaxf(x6, x7))));
    }

    const float wave_max = wave_reduce_max_64(local_max);
    if ((tid & 63) == 0)
        wave_maxes[tid >> 6] = wave_max;
    __syncthreads();

    float block_max = tid < (Threads / 64) ? wave_maxes[tid] : 0.0f;
    if (tid < 64)
        block_max = wave_reduce_max_64(block_max);
    if (tid == 0) {
        constexpr float quant_max = kInt8 ? 127.0f : 448.0f;
        row_scale = fmaxf(block_max, 1.0e-4f) / quant_max;
        out_scale[row] = row_scale;
    }
    __syncthreads();

    const float inv_scale = 1.0f / row_scale;
    for (int col = tid * 8; col < hidden; col += Threads * 8) {
        const uint64_t packed =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col);
        const uint64_t packed_hi =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 4);
        const float x0 = bf16_bits_to_float(static_cast<uint16_t>(packed)) * inv_scale;
        const float x1 = bf16_bits_to_float(static_cast<uint16_t>(packed >> 16)) * inv_scale;
        const float x2 = bf16_bits_to_float(static_cast<uint16_t>(packed >> 32)) * inv_scale;
        const float x3 = bf16_bits_to_float(static_cast<uint16_t>(packed >> 48)) * inv_scale;
        const float x4 = bf16_bits_to_float(static_cast<uint16_t>(packed_hi)) * inv_scale;
        const float x5 = bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 16)) * inv_scale;
        const float x6 = bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 32)) * inv_scale;
        const float x7 = bf16_bits_to_float(static_cast<uint16_t>(packed_hi >> 48)) * inv_scale;
        uint32_t* out_vec = reinterpret_cast<uint32_t*>(out_fp8 + x_row_offset + col);
        out_vec[0] = pack4_channelwise_quant<kInt8>(x0, x1, x2, x3);
        out_vec[1] = pack4_channelwise_quant<kInt8>(x4, x5, x6, x7);
    }
}

template <bool TopkIdxI64, bool TopkWeightsBf16>
__global__ __launch_bounds__(256)
void mega_moe_pre_dispatch_fp8_channelwise_vec16_4096_kernel(
    const uint16_t* __restrict__ x_bf16,
    const void* __restrict__ topk_idx,
    const void* __restrict__ topk_weights,
    uint8_t* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    int64_t* __restrict__ out_topk_idx,
    float* __restrict__ out_topk_weights,
    const int rows,
    const int topk) {
    constexpr int Threads = 256;
    constexpr int Hidden = 4096;
    constexpr int Vec = 16;
    __shared__ float wave_maxes[Threads / 64];
    __shared__ float row_scale;

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;

    const int tid = static_cast<int>(threadIdx.x);
    const int64_t x_row_offset = static_cast<int64_t>(row) * Hidden;
    const int64_t topk_row_offset = static_cast<int64_t>(row) * topk;

    if (tid < topk) {
        const int64_t route_offset = topk_row_offset + tid;
        stage_topk_route<TopkIdxI64, TopkWeightsBf16>(
            topk_idx, topk_weights, out_topk_idx, out_topk_weights, route_offset);
    }

    const int col = tid * Vec;
    const uint64_t packed0 =
        *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col);
    const uint64_t packed1 =
        *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 4);
    const uint64_t packed2 =
        *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 8);
    const uint64_t packed3 =
        *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 12);

    float values[Vec];
    values[0] = bf16_bits_to_float(static_cast<uint16_t>(packed0));
    values[1] = bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 16));
    values[2] = bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 32));
    values[3] = bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 48));
    values[4] = bf16_bits_to_float(static_cast<uint16_t>(packed1));
    values[5] = bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 16));
    values[6] = bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 32));
    values[7] = bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 48));
    values[8] = bf16_bits_to_float(static_cast<uint16_t>(packed2));
    values[9] = bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 16));
    values[10] = bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 32));
    values[11] = bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 48));
    values[12] = bf16_bits_to_float(static_cast<uint16_t>(packed3));
    values[13] = bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 16));
    values[14] = bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 32));
    values[15] = bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 48));

    float local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < Vec; ++i)
        local_max = fmaxf(local_max, fabsf(values[i]));

    const float wave_max = wave_reduce_max_64(local_max);
    if ((tid & 63) == 0)
        wave_maxes[tid >> 6] = wave_max;
    __syncthreads();

    float block_max = tid < (Threads / 64) ? wave_maxes[tid] : 0.0f;
    if (tid < 64)
        block_max = wave_reduce_max_64(block_max);
    if (tid == 0) {
        row_scale = fmaxf(block_max, 1.0e-4f) / 448.0f;
        out_scale[row] = row_scale;
    }
    __syncthreads();

    const float inv_scale = 1.0f / row_scale;
    uint32_t* out_vec = reinterpret_cast<uint32_t*>(out_fp8 + x_row_offset + col);
    out_vec[0] = pack4_fp8_e4m3fn(
        values[0] * inv_scale, values[1] * inv_scale,
        values[2] * inv_scale, values[3] * inv_scale);
    out_vec[1] = pack4_fp8_e4m3fn(
        values[4] * inv_scale, values[5] * inv_scale,
        values[6] * inv_scale, values[7] * inv_scale);
    out_vec[2] = pack4_fp8_e4m3fn(
        values[8] * inv_scale, values[9] * inv_scale,
        values[10] * inv_scale, values[11] * inv_scale);
    out_vec[3] = pack4_fp8_e4m3fn(
        values[12] * inv_scale, values[13] * inv_scale,
        values[14] * inv_scale, values[15] * inv_scale);
}

template <bool TopkIdxI64, bool TopkWeightsBf16, bool kInt8>
__global__ __launch_bounds__(256)
void mega_moe_pre_dispatch_fp8_channelwise_wave4_4096_kernel(
    const uint16_t* __restrict__ x_bf16,
    const void* __restrict__ topk_idx,
    const void* __restrict__ topk_weights,
    uint8_t* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    int64_t* __restrict__ out_topk_idx,
    float* __restrict__ out_topk_weights,
    const int rows,
    const int topk) {
    constexpr int Hidden = 4096;
    constexpr int Vec = 16;
    constexpr int Wave = 64;
    constexpr int TokensPerCta = 4;

    const int tid = static_cast<int>(threadIdx.x);
    const int wave_id = tid / Wave;
    const int lane = tid & (Wave - 1);
    const int row = static_cast<int>(blockIdx.x) * TokensPerCta + wave_id;
    if (row >= rows)
        return;

    const int64_t x_row_offset = static_cast<int64_t>(row) * Hidden;
    const int64_t topk_row_offset = static_cast<int64_t>(row) * topk;

    if (lane < topk) {
        const int64_t route_offset = topk_row_offset + lane;
        stage_topk_route<TopkIdxI64, TopkWeightsBf16>(
            topk_idx, topk_weights, out_topk_idx, out_topk_weights, route_offset);
    }

    float local_max = 0.0f;
    for (int col = lane * Vec; col < Hidden; col += Wave * Vec) {
        const uint64_t packed0 =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col);
        const uint64_t packed1 =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 4);
        const uint64_t packed2 =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 8);
        const uint64_t packed3 =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 12);
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed0))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 16))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 32))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 48))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed1))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 16))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 32))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 48))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed2))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 16))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 32))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 48))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed3))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 16))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 32))));
        local_max = fmaxf(local_max, fabsf(bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 48))));
    }

    const float row_max = wave_reduce_max_64(local_max);
    constexpr float quant_max = kInt8 ? 127.0f : 448.0f;
    const float row_scale = fmaxf(row_max, 1.0e-4f) / quant_max;
    if (lane == 0)
        out_scale[row] = row_scale;
    const float inv_scale = 1.0f / row_scale;

    for (int col = lane * Vec; col < Hidden; col += Wave * Vec) {
        const uint64_t packed0 =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col);
        const uint64_t packed1 =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 4);
        const uint64_t packed2 =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 8);
        const uint64_t packed3 =
            *reinterpret_cast<const uint64_t*>(x_bf16 + x_row_offset + col + 12);
        uint32_t* out_vec = reinterpret_cast<uint32_t*>(out_fp8 + x_row_offset + col);
        out_vec[0] = pack4_channelwise_quant<kInt8>(
            bf16_bits_to_float(static_cast<uint16_t>(packed0)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 16)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 32)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed0 >> 48)) * inv_scale);
        out_vec[1] = pack4_channelwise_quant<kInt8>(
            bf16_bits_to_float(static_cast<uint16_t>(packed1)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 16)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 32)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed1 >> 48)) * inv_scale);
        out_vec[2] = pack4_channelwise_quant<kInt8>(
            bf16_bits_to_float(static_cast<uint16_t>(packed2)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 16)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 32)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed2 >> 48)) * inv_scale);
        out_vec[3] = pack4_channelwise_quant<kInt8>(
            bf16_bits_to_float(static_cast<uint16_t>(packed3)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 16)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 32)) * inv_scale,
            bf16_bits_to_float(static_cast<uint16_t>(packed3 >> 48)) * inv_scale);
    }
}

__global__ void deepep_scatter_prefix_kernel(const int* num_recv_tokens_per_expert,
                                             int* expert_start_loc,
                                             const int num_experts) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    int offset = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
        expert_start_loc[expert] = offset;
        offset += num_recv_tokens_per_expert[expert];
    }
}

template <typename TopkT>
__global__ void deepep_scatter_channelwise_kernel(uint8_t* grouped_x,
                                                  float* grouped_x_scale,
                                                  float* route_weights,
                                                  int* m_indices,
                                                  int* output_index,
                                                  int* expert_start_loc,
                                                  const uint8_t* recv_x,
                                                  const float* recv_x_scale,
                                                  const TopkT* recv_topk_ids,
                                                  const float* recv_topk_weights,
                                                  const int total_rows,
                                                  const int recv_rows,
                                                  const int topk,
                                                  const int hidden,
                                                  const int num_experts) {
    const int route = static_cast<int>(blockIdx.x);
    if (route >= recv_rows * topk)
        return;

    const int token = route / topk;
    const int slot = route - token * topk;
    const int expert = static_cast<int>(recv_topk_ids[route]);
    __shared__ int dst_row;
    if (threadIdx.x == 0) {
        if (expert >= 0 && expert < num_experts) {
            const int row = atomicAdd(expert_start_loc + expert, 1);
            dst_row = row < total_rows ? row : -1;
            output_index[route] = dst_row;
            if (dst_row >= 0) {
                grouped_x_scale[dst_row] = recv_x_scale[token];
                route_weights[dst_row] = recv_topk_weights[route];
                m_indices[dst_row] = expert;
            }
        } else {
            dst_row = -1;
            output_index[route] = -1;
        }
    }
    __syncthreads();

    if (dst_row < 0)
        return;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        grouped_x[static_cast<int64_t>(dst_row) * hidden + h] =
            recv_x[static_cast<int64_t>(token) * hidden + h];
    }
}

template <typename TopkT>
__global__ void deepep_gather_channelwise_kernel(uint16_t* recv_y,
                                                 const uint16_t* l2_out,
                                                 const TopkT* recv_topk_ids,
                                                 const float* recv_topk_weights,
                                                 const int* output_index,
                                                 const int recv_rows,
                                                 const int topk,
                                                 const int hidden,
                                                 const bool apply_topk_weights) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= recv_rows)
        return;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float sum = 0.0f;
        for (int slot = 0; slot < topk; ++slot) {
            const int route = token * topk + slot;
            const int expert = static_cast<int>(recv_topk_ids[route]);
            const int src_row = output_index[route];
            if (expert >= 0 && src_row >= 0) {
                float value = bf16_bits_to_float(l2_out[static_cast<int64_t>(src_row) * hidden + h]);
                if (apply_topk_weights)
                    value *= recv_topk_weights[route];
                sum += value;
            }
        }
        recv_y[static_cast<int64_t>(token) * hidden + h] = float_to_bf16_bits(sum);
    }
}

__global__ void deepep_sanitize_int8_tile_heads_kernel(float* grouped_x_scale,
                                                       float* route_weights,
                                                       int* m_indices,
                                                       const int total_rows) {
    constexpr int kContiguousTileRows = 256;
    const int tile = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int tile_head = tile * kContiguousTileRows;
    if (tile_head < total_rows && m_indices[tile_head] < 0) {
        // The INT8 contiguous ASM dereferences the tile-head expert before its
        // per-row -1 mask. Point an empty tile at a valid zero-scaled dummy row.
        grouped_x_scale[tile_head] = 0.0f;
        route_weights[tile_head] = 0.0f;
        m_indices[tile_head] = 0;
    }
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
    const int total_rows,
    const int recv_rows,
    const int topk,
    const int hidden,
    const int num_experts,
    const bool topk_ids_i64,
    const bool sanitize_int8_tile_heads) {
    constexpr int kThreads = 256;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(deepep_scatter_prefix_kernel, dim3(1), dim3(1), 0, stream,
                       num_recv_tokens_per_expert, expert_start_loc, num_experts);
    const int blocks = std::max(1, recv_rows * topk);
    if (topk_ids_i64) {
        hipLaunchKernelGGL((deepep_scatter_channelwise_kernel<int64_t>),
                           dim3(blocks), dim3(kThreads), 0, stream,
                           static_cast<uint8_t*>(grouped_x),
                           grouped_x_scale,
                           route_weights,
                           m_indices,
                           output_index,
                           expert_start_loc,
                           static_cast<const uint8_t*>(recv_x),
                           recv_x_scale,
                           static_cast<const int64_t*>(recv_topk_ids),
                           recv_topk_weights,
                           total_rows,
                           recv_rows,
                           topk,
                           hidden,
                           num_experts);
    } else {
        hipLaunchKernelGGL((deepep_scatter_channelwise_kernel<int>),
                           dim3(blocks), dim3(kThreads), 0, stream,
                           static_cast<uint8_t*>(grouped_x),
                           grouped_x_scale,
                           route_weights,
                           m_indices,
                           output_index,
                           expert_start_loc,
                           static_cast<const uint8_t*>(recv_x),
                           recv_x_scale,
                           static_cast<const int*>(recv_topk_ids),
                           recv_topk_weights,
                           total_rows,
                           recv_rows,
                           topk,
                           hidden,
                           num_experts);
    }
    if (sanitize_int8_tile_heads) {
        constexpr int kContiguousTileRows = 256;
        const int tile_count = (total_rows + kContiguousTileRows - 1) /
                               kContiguousTileRows;
        const int sanitize_blocks = (tile_count + kThreads - 1) / kThreads;
        hipLaunchKernelGGL(deepep_sanitize_int8_tile_heads_kernel,
                           dim3(sanitize_blocks), dim3(kThreads), 0, stream,
                           grouped_x_scale,
                           route_weights,
                           m_indices,
                           total_rows);
    }
    DG_HIP_CHECK(hipGetLastError());
}

template <bool kInt8>
void launch_mega_moe_pre_dispatch_channelwise_hip_impl(
    const void* x_bf16,
    const void* topk_idx,
    const void* topk_weights,
    void* out_fp8,
    float* out_scale,
    int64_t* out_topk_idx,
    float* out_topk_weights,
    const int rows,
    const int hidden,
    const int topk,
    const bool topk_idx_i64,
    const bool topk_weights_bf16) {
    constexpr int kThreads = 256;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 grid(std::max(1, rows));
    const dim3 block(kThreads);
#define LAUNCH_PREDISPATCH_KERNEL(TOPK_I64, TOPK_W_BF16)                                      \
    do {                                                                                       \
        if (hidden == 4096) {                                                                   \
            hipLaunchKernelGGL(                                                                \
                (mega_moe_pre_dispatch_fp8_channelwise_wave4_4096_kernel<                     \
                    TOPK_I64, TOPK_W_BF16, kInt8>),                                            \
                dim3((rows + 3) / 4),                                                          \
                block,                                                                         \
                0,                                                                             \
                stream,                                                                        \
                static_cast<const uint16_t*>(x_bf16),                                          \
                topk_idx,                                                                      \
                topk_weights,                                                                  \
                static_cast<uint8_t*>(out_fp8),                                                \
                out_scale,                                                                     \
                out_topk_idx,                                                                  \
                out_topk_weights,                                                              \
                rows,                                                                          \
                topk);                                                                         \
        } else {                                                                               \
            hipLaunchKernelGGL(                                                                \
                (mega_moe_pre_dispatch_fp8_channelwise_kernel<                                 \
                    kThreads, TOPK_I64, TOPK_W_BF16, kInt8>),                                  \
                grid,                                                                          \
                block,                                                                         \
                0,                                                                             \
                stream,                                                                        \
                static_cast<const uint16_t*>(x_bf16),                                          \
                topk_idx,                                                                      \
                topk_weights,                                                                  \
                static_cast<uint8_t*>(out_fp8),                                                \
                out_scale,                                                                     \
                out_topk_idx,                                                                  \
                out_topk_weights,                                                              \
                rows,                                                                          \
                hidden,                                                                        \
                topk);                                                                         \
        }                                                                                      \
    } while (0)

    if (topk_idx_i64) {
        if (topk_weights_bf16) {
            LAUNCH_PREDISPATCH_KERNEL(true, true);
        } else {
            LAUNCH_PREDISPATCH_KERNEL(true, false);
        }
    } else {
        if (topk_weights_bf16) {
            LAUNCH_PREDISPATCH_KERNEL(false, true);
        } else {
            LAUNCH_PREDISPATCH_KERNEL(false, false);
        }
    }
#undef LAUNCH_PREDISPATCH_KERNEL
    DG_HIP_CHECK(hipGetLastError());
}

#define MEGA_MOE_PREDISPATCH_LAUNCH_ARGS                                      \
    const void* x_bf16, const void* topk_idx, const void* topk_weights,        \
    void* out_quant, float* out_scale, int64_t* out_topk_idx,                  \
    float* out_topk_weights, const int rows, const int hidden, const int topk, \
    const bool topk_idx_i64, const bool topk_weights_bf16

void launch_mega_moe_pre_dispatch_fp8_channelwise_hip(
    MEGA_MOE_PREDISPATCH_LAUNCH_ARGS) {
    launch_mega_moe_pre_dispatch_channelwise_hip_impl<false>(
        x_bf16, topk_idx, topk_weights, out_quant, out_scale, out_topk_idx,
        out_topk_weights, rows, hidden, topk, topk_idx_i64,
        topk_weights_bf16);
}

void launch_mega_moe_pre_dispatch_int8_channelwise_hip(
    MEGA_MOE_PREDISPATCH_LAUNCH_ARGS) {
    launch_mega_moe_pre_dispatch_channelwise_hip_impl<true>(
        x_bf16, topk_idx, topk_weights, out_quant, out_scale, out_topk_idx,
        out_topk_weights, rows, hidden, topk, topk_idx_i64,
        topk_weights_bf16);
}

#undef MEGA_MOE_PREDISPATCH_LAUNCH_ARGS

void launch_mega_moe_deepep_gather_channelwise_hip(
    void* recv_y,
    const void* l2_out,
    const void* recv_topk_ids,
    const float* recv_topk_weights,
    const int* output_index,
    const int recv_rows,
    const int topk,
    const int hidden,
    const bool topk_ids_i64,
    const bool apply_topk_weights) {
    constexpr int kThreads = 256;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int blocks = std::max(1, recv_rows);
    if (topk_ids_i64) {
        hipLaunchKernelGGL((deepep_gather_channelwise_kernel<int64_t>),
                           dim3(blocks), dim3(kThreads), 0, stream,
                           static_cast<uint16_t*>(recv_y),
                           static_cast<const uint16_t*>(l2_out),
                           static_cast<const int64_t*>(recv_topk_ids),
                           recv_topk_weights,
                           output_index,
                           recv_rows,
                           topk,
                           hidden,
                           apply_topk_weights);
    } else {
        hipLaunchKernelGGL((deepep_gather_channelwise_kernel<int>),
                           dim3(blocks), dim3(kThreads), 0, stream,
                           static_cast<uint16_t*>(recv_y),
                           static_cast<const uint16_t*>(l2_out),
                           static_cast<const int*>(recv_topk_ids),
                           recv_topk_weights,
                           output_index,
                           recv_rows,
                           topk,
                           hidden,
                           apply_topk_weights);
    }
    DG_HIP_CHECK(hipGetLastError());
}

} // namespace deep_gemm::mega
