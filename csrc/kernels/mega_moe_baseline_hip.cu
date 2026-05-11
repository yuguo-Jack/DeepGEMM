#include <algorithm>
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
    const bool topk_ids_i64) {
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
    DG_HIP_CHECK(hipGetLastError());
}

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
