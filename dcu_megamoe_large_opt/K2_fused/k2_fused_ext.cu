#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

template <int Threads>
__global__ __launch_bounds__(Threads)
void swiglu_quant_channelwise_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ topk_weights,
    uint8_t* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    uint16_t* __restrict__ out_bf16,
    const int rows,
    const int hidden,
    const bool has_topk_weights,
    const bool has_clamp_value,
    const bool output_bf16,
    const float clamp_value) {
    extern __shared__ float smem[];
    float* y_smem = smem;
    float* reduce_smem = smem + hidden;

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;

    const int tid = static_cast<int>(threadIdx.x);
    const int stride = hidden * 2;
    const int64_t row_base = static_cast<int64_t>(row) * stride;
    const float route_weight = has_topk_weights ? topk_weights[row] : 1.0f;

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
            const float sig = 1.0f / (1.0f + __expf(-gate));
            const float y = gate * sig * up * route_weight;
            y_smem[col + i] = y;
            local_amax = fmaxf(local_amax, fabsf(y));
        }
    }

    reduce_smem[tid] = local_amax;
    __syncthreads();

    for (int offset = Threads / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            reduce_smem[tid] = fmaxf(reduce_smem[tid], reduce_smem[tid + offset]);
        __syncthreads();
    }

    const float clamped_amax = fmaxf(reduce_smem[0], 1.0e-4f);
    const float scale = clamped_amax / 448.0f;
    const float inv_scale = 448.0f / clamped_amax;
    if (tid == 0)
        out_scale[row] = scale;

    const int64_t out_base = static_cast<int64_t>(row) * hidden;
    for (int col = tid * 4; col < hidden; col += Threads * 4) {
        const float y0 = y_smem[col + 0];
        const float y1 = y_smem[col + 1];
        const float y2 = y_smem[col + 2];
        const float y3 = y_smem[col + 3];
        *reinterpret_cast<uint32_t*>(out_fp8 + out_base + col) =
            pack4_e4m3fn(y0 * inv_scale, y1 * inv_scale, y2 * inv_scale, y3 * inv_scale);
        if (output_bf16) {
            const uint32_t b01 =
                static_cast<uint32_t>(float_to_bf16_bits(y0)) |
                (static_cast<uint32_t>(float_to_bf16_bits(y1)) << 16);
            const uint32_t b23 =
                static_cast<uint32_t>(float_to_bf16_bits(y2)) |
                (static_cast<uint32_t>(float_to_bf16_bits(y3)) << 16);
            uint32_t* bf16_ptr = reinterpret_cast<uint32_t*>(out_bf16 + out_base + col);
            bf16_ptr[0] = b01;
            bf16_ptr[1] = b23;
        }
    }
}

template <int Threads, int VecGroups>
__global__ __launch_bounds__(Threads)
void swiglu_quant_channelwise_reg_kernel(
    const uint16_t* __restrict__ x,
    const float* __restrict__ topk_weights,
    uint8_t* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    uint16_t* __restrict__ out_bf16,
    const int rows,
    const bool has_topk_weights,
    const bool has_clamp_value,
    const bool output_bf16,
    const float clamp_value) {
    static_assert(VecGroups > 0, "VecGroups must be positive");
    constexpr int hidden = Threads * VecGroups * 4;
    extern __shared__ float reduce_smem[];

    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;

    const int tid = static_cast<int>(threadIdx.x);
    const int stride = hidden * 2;
    const int64_t row_base = static_cast<int64_t>(row) * stride;
    const float route_weight = has_topk_weights ? topk_weights[row] : 1.0f;

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
        y0[g] = gate0 * (1.0f / (1.0f + __expf(-gate0))) * up0 * route_weight;
        y1[g] = gate1 * (1.0f / (1.0f + __expf(-gate1))) * up1 * route_weight;
        y2[g] = gate2 * (1.0f / (1.0f + __expf(-gate2))) * up2 * route_weight;
        y3[g] = gate3 * (1.0f / (1.0f + __expf(-gate3))) * up3 * route_weight;
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
    const float inv_scale = 448.0f / clamped_amax;
    if (tid == 0)
        out_scale[row] = clamped_amax / 448.0f;

    const int64_t out_base = static_cast<int64_t>(row) * hidden;
#pragma unroll
    for (int g = 0; g < VecGroups; ++g) {
        const int col = (g * Threads + tid) * 4;
        *reinterpret_cast<uint32_t*>(out_fp8 + out_base + col) =
            pack4_e4m3fn(y0[g] * inv_scale, y1[g] * inv_scale,
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
}

template <int Threads>
void launch_swiglu_quant_channelwise(
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    torch::Tensor& out_fp8,
    torch::Tensor& out_scale,
    torch::Tensor& out_bf16,
    const bool output_bf16,
    const bool has_clamp_value,
    const double clamp_value) {
    const int rows = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1) / 2);
    if (rows == 0)
        return;
    const bool has_topk_weights = topk_weights.numel() > 0;
    const size_t shared_bytes_for_reg =
        Threads == 64 ? 0 : static_cast<size_t>(Threads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (hidden == 2048) {
        constexpr int vec_groups = 2048 / (Threads * 4);
        hipLaunchKernelGGL(
            (swiglu_quant_channelwise_reg_kernel<Threads, vec_groups>),
            dim3(rows),
            dim3(Threads),
            shared_bytes_for_reg,
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
            has_topk_weights ? topk_weights.data_ptr<float>() : nullptr,
            reinterpret_cast<uint8_t*>(out_fp8.data_ptr()),
            out_scale.data_ptr<float>(),
            output_bf16 ? reinterpret_cast<uint16_t*>(out_bf16.data_ptr<at::BFloat16>()) : nullptr,
            rows,
            has_topk_weights,
            has_clamp_value,
            output_bf16,
            static_cast<float>(clamp_value));
    } else if (hidden == 4096) {
        constexpr int vec_groups = 4096 / (Threads * 4);
        hipLaunchKernelGGL(
            (swiglu_quant_channelwise_reg_kernel<Threads, vec_groups>),
            dim3(rows),
            dim3(Threads),
            shared_bytes_for_reg,
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
            has_topk_weights ? topk_weights.data_ptr<float>() : nullptr,
            reinterpret_cast<uint8_t*>(out_fp8.data_ptr()),
            out_scale.data_ptr<float>(),
            output_bf16 ? reinterpret_cast<uint16_t*>(out_bf16.data_ptr<at::BFloat16>()) : nullptr,
            rows,
            has_topk_weights,
            has_clamp_value,
            output_bf16,
            static_cast<float>(clamp_value));
    } else {
        const size_t shared_bytes =
            static_cast<size_t>(hidden + Threads) * sizeof(float);
        hipLaunchKernelGGL(
            (swiglu_quant_channelwise_kernel<Threads>),
            dim3(rows),
            dim3(Threads),
            shared_bytes,
            stream,
            reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()),
            has_topk_weights ? topk_weights.data_ptr<float>() : nullptr,
            reinterpret_cast<uint8_t*>(out_fp8.data_ptr()),
            out_scale.data_ptr<float>(),
            output_bf16 ? reinterpret_cast<uint16_t*>(out_bf16.data_ptr<at::BFloat16>()) : nullptr,
            rows,
            hidden,
            has_topk_weights,
            has_clamp_value,
            output_bf16,
            static_cast<float>(clamp_value));
    }
    K2_HIP_CHECK(hipGetLastError());
}

void launch_swiglu_quant_channelwise_auto(
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    torch::Tensor& out_fp8,
    torch::Tensor& out_scale,
    torch::Tensor& out_bf16,
    const bool output_bf16,
    const bool has_clamp_value,
    const double clamp_value) {
    const int hidden = static_cast<int>(x.size(1) / 2);
    if (hidden <= 2048 && !output_bf16) {
        launch_swiglu_quant_channelwise<64>(
            x, topk_weights, out_fp8, out_scale, out_bf16,
            output_bf16, has_clamp_value, clamp_value);
    } else if (hidden <= 2048) {
        launch_swiglu_quant_channelwise<128>(
            x, topk_weights, out_fp8, out_scale, out_bf16,
            output_bf16, has_clamp_value, clamp_value);
    } else if (hidden == 4096) {
        launch_swiglu_quant_channelwise<128>(
            x, topk_weights, out_fp8, out_scale, out_bf16,
            output_bf16, has_clamp_value, clamp_value);
    } else {
        launch_swiglu_quant_channelwise<256>(
            x, topk_weights, out_fp8, out_scale, out_bf16,
            output_bf16, has_clamp_value, clamp_value);
    }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA/HIP");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
swiglu_quant_channelwise(
    const torch::Tensor& x,
    const torch::Tensor& topk_weights,
    const bool output_bf16,
    const bool has_clamp_value,
    const double clamp_value) {
    check_cuda_contiguous(x, "x");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(x.dim() == 2, "x must be [rows, 2 * hidden]");
    TORCH_CHECK(x.size(1) % 2 == 0, "x hidden dimension must be even");
    const int64_t rows = x.size(0);
    const int64_t hidden = x.size(1) / 2;
    TORCH_CHECK(hidden > 0 && hidden <= 4096,
                "K2 fused currently supports hidden in (0, 4096]");
    TORCH_CHECK(hidden % 4 == 0,
                "K2 fused vectorized path requires hidden to be divisible by 4");
    if (topk_weights.numel() > 0) {
        check_cuda_contiguous(topk_weights, "topk_weights");
        TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32,
                    "topk_weights must be FP32");
        TORCH_CHECK(topk_weights.numel() >= rows,
                    "topk_weights must cover every row");
    }

    auto out_fp8 = torch::empty(
        {rows, hidden},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(x.device()));
    auto out_scale = torch::empty(
        {rows},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    auto out_bf16 = output_bf16
        ? torch::empty(
              {rows, hidden},
              torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()))
        : torch::empty(
              {0},
              torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));

    launch_swiglu_quant_channelwise_auto(
        x, topk_weights, out_fp8, out_scale, out_bf16,
        output_bf16, has_clamp_value, clamp_value);

    return {out_fp8, out_scale, out_bf16};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_quant_channelwise", &swiglu_quant_channelwise,
          pybind11::arg("x"),
          pybind11::arg("topk_weights"),
          pybind11::arg("output_bf16") = false,
          pybind11::arg("has_clamp_value") = true,
          pybind11::arg("clamp_value") = 10.0);
}
