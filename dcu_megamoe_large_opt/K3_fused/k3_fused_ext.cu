#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include <deep_gemm/common/mega_moe_dcu.cuh>
#include <deep_gemm/comm/mega_moe_dcu.cuh>
#include <deep_gemm/layout/mega_moe_dcu.cuh>

#define K3_HIP_CHECK(expr)                                                       \
    do {                                                                        \
        const hipError_t _status = (expr);                                      \
        TORCH_CHECK(_status == hipSuccess, #expr " failed: ",                  \
                    hipGetErrorString(_status));                                \
    } while (0)

namespace {

static constexpr const char* kAsmKernelName =
    "DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16";
static constexpr const char* kCombineAsmKernelName =
    "DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE";
static constexpr int64_t kAsmRowPointerPadding = 512;

struct __attribute__((packed)) GpuProb {
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t k;
    void* d;
    void* c;
    void* a;
    void* b;
    uint32_t strideD1;
    uint32_t strideD2;
    uint32_t strideC1;
    uint32_t strideC2;
    uint32_t strideA1;
    uint32_t strideA2;
    uint32_t strideB1;
    uint32_t strideB2;
    int8_t alpha[16];
    int8_t beta[16];
    float* scaleA;
    float* scaleB;
    int32_t* asm_done_counter;
    int64_t* asm_signal_addrs;
    uint32_t asm_done_target;
    uint32_t asm_signal_num_ranks;
    uint32_t asm_tail_signal;
    uint32_t asm_tail_reserved;
    uint16_t* asm_reduce_y;
    uint16_t* asm_reduce_combine;
    uint32_t asm_reduce_total_vecs;
    uint32_t asm_reduce_slot_stride_vec;
    uint32_t asm_signal_generation;
    uint32_t asm_tail_reduce;
    uint32_t asm_reduce_blocks;
};

static_assert(offsetof(GpuProb, scaleA) == 0x70,
              "asm expects scaleA at GpuProb+0x70");
static_assert(offsetof(GpuProb, scaleB) == 0x78,
              "asm expects scaleB at GpuProb+0x78");
static_assert(offsetof(GpuProb, asm_done_counter) == 0x80,
              "K3 tail signal asm expects done counter at GpuProb+0x80");
static_assert(offsetof(GpuProb, asm_signal_addrs) == 0x88,
              "K3 tail signal asm expects signal addrs at GpuProb+0x88");
static_assert(offsetof(GpuProb, asm_done_target) == 0x90,
              "K3 tail signal asm expects done target at GpuProb+0x90");
static_assert(offsetof(GpuProb, asm_reduce_y) == 0xa0,
              "K3 tail reduce asm expects y at GpuProb+0xa0");
static_assert(offsetof(GpuProb, asm_reduce_combine) == 0xa8,
              "K3 tail reduce asm expects combine at GpuProb+0xa8");
static_assert(offsetof(GpuProb, asm_reduce_total_vecs) == 0xb0,
              "K3 tail reduce asm expects total vecs at GpuProb+0xb0");
static_assert(offsetof(GpuProb, asm_reduce_blocks) == 0xc0,
              "K3 tail reduce asm expects reduce blocks at GpuProb+0xc0");

struct __attribute__((packed)) KernelArgs {
    uint32_t gemm_count;
    void const* DeviceUserArguments;
    void const* argsPtr;
    uint32_t kipWgTableGen;
    uint32_t gsu;
    int32_t* m_indics;
    int32_t* debug_d;
};

struct LoadedAsmKernel {
    std::mutex mutex;
    std::string path;
    std::string kernel_name;
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
};

LoadedAsmKernel& asm_kernel_cache() {
    static LoadedAsmKernel cache;
    return cache;
}

hipFunction_t get_asm_function(const std::string& code_object_path,
                               const char* kernel_name) {
    auto& cache = asm_kernel_cache();
    std::lock_guard<std::mutex> guard(cache.mutex);
    if (cache.module != nullptr && cache.path == code_object_path &&
        cache.kernel_name == kernel_name) {
        return cache.function;
    }
    if (cache.module != nullptr) {
        K3_HIP_CHECK(hipModuleUnload(cache.module));
        cache.module = nullptr;
        cache.function = nullptr;
        cache.path.clear();
        cache.kernel_name.clear();
    }
    K3_HIP_CHECK(hipModuleLoad(&cache.module, code_object_path.c_str()));
    K3_HIP_CHECK(hipModuleGetFunction(&cache.function, cache.module, kernel_name));
    cache.path = code_object_path;
    cache.kernel_name = kernel_name;
    return cache.function;
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA/HIP");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int env_int(const char* name, const int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return fallback;
    return std::atoi(value);
}

void launch_l2_deepgemm_original_asm(
    const torch::Tensor& output,
    const torch::Tensor& act_fp8,
    const torch::Tensor& act_scale,
    const torch::Tensor& m_indices,
    const torch::Tensor& l2_weight,
    const torch::Tensor& l2_scale,
    const std::string& code_object_path,
    const char* kernel_name,
    const torch::Tensor* row_combine_ptrs,
    const torch::Tensor* asm_done_counter = nullptr,
    const torch::Tensor* asm_signal_addrs = nullptr,
    const int64_t asm_done_target = 0,
    const int64_t asm_signal_num_ranks = 0,
    const bool asm_tail_signal = false,
    const torch::Tensor* asm_reduce_y = nullptr,
    const torch::Tensor* sym_buffer = nullptr,
    const int64_t asm_signal_generation = 0,
    const int64_t reduce_num_ranks = 0,
    const int64_t reduce_num_experts = 0,
    const int64_t reduce_num_max_tokens_per_rank = 0,
    const int64_t reduce_num_tokens = 0,
    const int64_t reduce_num_topk = 0,
    const int64_t reduce_hidden = 0) {
    check_cuda_contiguous(output, "output");
    check_cuda_contiguous(act_fp8, "act_fp8");
    check_cuda_contiguous(act_scale, "act_scale");
    check_cuda_contiguous(m_indices, "m_indices");
    check_cuda_contiguous(l2_weight, "l2_weight");
    check_cuda_contiguous(l2_scale, "l2_scale");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
                "output must be BF16");
    TORCH_CHECK(act_fp8.scalar_type() == torch::kFloat8_e4m3fn,
                "act_fp8 must be FP8 E4M3");
    TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32,
                "act_scale must be FP32");
    TORCH_CHECK(m_indices.scalar_type() == torch::kInt,
                "m_indices must be int32");
    TORCH_CHECK(l2_weight.scalar_type() == torch::kFloat8_e4m3fn,
                "l2_weight must be FP8 E4M3");
    TORCH_CHECK(l2_scale.scalar_type() == torch::kFloat32,
                "l2_scale must be FP32");

    const int total_rows = static_cast<int>(act_fp8.size(0));
    const int k = static_cast<int>(act_fp8.size(1));
    const int hidden = static_cast<int>(output.size(1));
    const int local_experts = static_cast<int>(l2_weight.size(0));
    TORCH_CHECK(output.dim() == 2 && output.size(0) == total_rows,
                "output shape must be [rows, hidden]");
    TORCH_CHECK(act_scale.numel() == total_rows,
                "act_scale must have one channelwise scale per row");
    TORCH_CHECK(m_indices.numel() >= total_rows,
                "m_indices length must cover total rows");
    TORCH_CHECK(total_rows > 0 && total_rows % 256 == 0,
                "K3 asm expects rows padded to a 256-row tile");
    TORCH_CHECK(hidden == 4096 && k == 2048,
                "current K3 asm path is specialized for hidden=4096, intermediate=2048");
    TORCH_CHECK(l2_weight.dim() == 3 &&
                    l2_weight.size(1) == hidden / 16 &&
                    l2_weight.size(2) == k * 16,
                "invalid Marlin L2 weight shape");
    TORCH_CHECK(l2_scale.dim() == 2 &&
                    l2_scale.size(0) == local_experts &&
                    l2_scale.size(1) == hidden,
                "invalid L2 scale shape");
    if (row_combine_ptrs != nullptr) {
        check_cuda_contiguous(*row_combine_ptrs, "row_combine_ptrs");
        TORCH_CHECK(row_combine_ptrs->scalar_type() == torch::kInt64,
                    "row_combine_ptrs must be int64");
        TORCH_CHECK(row_combine_ptrs->numel() >= total_rows,
                    "row_combine_ptrs must have one pointer per row");
    }
    if (asm_tail_signal) {
        TORCH_CHECK(row_combine_ptrs != nullptr,
                    "K3 asm tail signal is only supported by combine asm");
        TORCH_CHECK(asm_done_counter != nullptr && asm_signal_addrs != nullptr,
                    "asm_done_counter and asm_signal_addrs are required");
        check_cuda_contiguous(*asm_done_counter, "asm_done_counter");
        check_cuda_contiguous(*asm_signal_addrs, "asm_signal_addrs");
        TORCH_CHECK(asm_done_counter->scalar_type() == torch::kInt,
                    "asm_done_counter must be int32");
        TORCH_CHECK(asm_done_counter->numel() >= 1,
                    "asm_done_counter must have at least one element");
        TORCH_CHECK(asm_signal_addrs->scalar_type() == torch::kInt64,
                    "asm_signal_addrs must be int64");
        TORCH_CHECK(asm_signal_addrs->numel() >= 16,
                    "asm_signal_addrs must contain 8 local and 8 peer signal addresses");
        TORCH_CHECK(asm_signal_num_ranks > 0 && asm_signal_num_ranks <= 8,
                    "asm_signal_num_ranks must be in [1, 8]");
        TORCH_CHECK(asm_done_target > 0 &&
                        asm_done_target <= static_cast<int64_t>(UINT32_MAX),
                    "asm_done_target must fit in uint32");
    }
    if (asm_reduce_y != nullptr) {
        TORCH_CHECK(asm_tail_signal,
                    "asm tail reduce requires asm_tail_signal");
        TORCH_CHECK(sym_buffer != nullptr,
                    "sym_buffer is required for asm tail reduce");
        check_cuda_contiguous(*asm_reduce_y, "asm_reduce_y");
        check_cuda_contiguous(*sym_buffer, "sym_buffer");
        TORCH_CHECK(asm_reduce_y->scalar_type() == torch::kBFloat16,
                    "asm_reduce_y must be BF16");
        TORCH_CHECK(sym_buffer->scalar_type() == torch::kInt8,
                    "sym_buffer must be int8");
        TORCH_CHECK(reduce_hidden == hidden && reduce_hidden == 4096 &&
                        reduce_num_topk == 6,
                    "asm tail reduce currently supports hidden=4096, topk=6");
        TORCH_CHECK(asm_reduce_y->dim() == 2 &&
                        asm_reduce_y->size(0) == reduce_num_tokens &&
                        asm_reduce_y->size(1) == reduce_hidden,
                    "asm_reduce_y shape mismatch");
        TORCH_CHECK(reduce_num_tokens > 0 &&
                        reduce_num_max_tokens_per_rank >= reduce_num_tokens,
                    "invalid reduce token counts");
        const int64_t total_reduce_vecs =
            reduce_num_tokens * (reduce_hidden / 8);
        const int64_t slot_stride_vec =
            reduce_num_max_tokens_per_rank * (reduce_hidden / 8);
        TORCH_CHECK(total_reduce_vecs <= static_cast<int64_t>(UINT32_MAX) &&
                        slot_stride_vec <= static_cast<int64_t>(UINT32_MAX),
                    "asm tail reduce vector counts must fit in uint32");
    }

    GpuProb prob{};
    prob.m = static_cast<uint32_t>(hidden);
    prob.n = static_cast<uint32_t>(total_rows);
    prob.batch = 1;
    prob.k = static_cast<uint32_t>(k);
    prob.d = row_combine_ptrs == nullptr
        ? output.data_ptr()
        : row_combine_ptrs->data_ptr<int64_t>();
    prob.c = output.data_ptr();
    prob.a = l2_weight.data_ptr();
    prob.b = act_fp8.data_ptr();
    prob.strideD1 = static_cast<uint32_t>(hidden);
    prob.strideD2 = static_cast<uint32_t>(total_rows * hidden);
    prob.strideC1 = static_cast<uint32_t>(hidden);
    prob.strideC2 = static_cast<uint32_t>(total_rows * hidden);
    prob.strideA1 = static_cast<uint32_t>(k);
    prob.strideA2 = static_cast<uint32_t>(hidden * k);
    prob.strideB1 = static_cast<uint32_t>(k);
    prob.strideB2 = static_cast<uint32_t>(total_rows * k);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::memcpy(prob.alpha, &alpha, sizeof(float));
    std::memcpy(prob.beta, &beta, sizeof(float));
    prob.scaleA = l2_scale.data_ptr<float>();
    prob.scaleB = act_scale.data_ptr<float>();
    if (asm_tail_signal) {
        prob.asm_done_counter = asm_done_counter->data_ptr<int32_t>();
        prob.asm_signal_addrs = asm_signal_addrs->data_ptr<int64_t>();
        prob.asm_done_target = static_cast<uint32_t>(asm_done_target);
        prob.asm_signal_num_ranks = static_cast<uint32_t>(asm_signal_num_ranks);
        prob.asm_tail_signal = 1;
    }
    if (asm_reduce_y != nullptr) {
        const int64_t combine_offset = deep_gemm::mega::combine_token_offset(
            static_cast<int>(reduce_num_ranks),
            static_cast<int>(reduce_num_experts),
            static_cast<int>(reduce_num_max_tokens_per_rank),
            static_cast<int>(reduce_num_topk),
            static_cast<int>(reduce_hidden));
        const int reduce_blocks =
            std::clamp(env_int("K3_ASM_TAIL_REDUCE_BLOCKS", 128), 0, 128);
        prob.asm_reduce_y = reinterpret_cast<uint16_t*>(asm_reduce_y->data_ptr());
        prob.asm_reduce_combine = reinterpret_cast<uint16_t*>(
            reinterpret_cast<uint8_t*>(sym_buffer->data_ptr<int8_t>()) + combine_offset);
        prob.asm_reduce_total_vecs = static_cast<uint32_t>(
            reduce_num_tokens * (reduce_hidden / 8));
        prob.asm_reduce_slot_stride_vec = static_cast<uint32_t>(
            reduce_num_max_tokens_per_rank * (reduce_hidden / 8));
        prob.asm_signal_generation = static_cast<uint32_t>(asm_signal_generation);
        prob.asm_tail_reduce = 1;
        prob.asm_reduce_blocks = static_cast<uint32_t>(reduce_blocks);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto prob_storage = torch::empty(
        {static_cast<int64_t>(sizeof(GpuProb))},
        torch::TensorOptions().dtype(torch::kUInt8).device(output.device()));
    void* prob_device = prob_storage.data_ptr();
    K3_HIP_CHECK(hipMemcpyAsync(
        prob_device, &prob, sizeof(GpuProb), hipMemcpyHostToDevice, stream));

    KernelArgs args{};
    args.gemm_count = 1;
    args.DeviceUserArguments = prob_device;
    args.argsPtr = nullptr;
    args.kipWgTableGen = 0;
    args.gsu = 1;
    args.m_indics = m_indices.data_ptr<int32_t>();
    args.debug_d = row_combine_ptrs == nullptr
        ? nullptr
        : reinterpret_cast<int32_t*>(row_combine_ptrs->data_ptr<int64_t>());

    const int local_work_size = 768;
    const int wg_m = (hidden + 255) / 256;
    const int wg_n = (total_rows + 255) / 256;
    const int gemm_workgroups = wg_m * wg_n;
    const int reduce_workgroups = static_cast<int>(prob.asm_reduce_blocks);
    const size_t global_work_items =
        static_cast<size_t>(local_work_size) *
        static_cast<size_t>(gemm_workgroups + reduce_workgroups);

    hipFunction_t function = get_asm_function(code_object_path, kernel_name);
    size_t arg_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END};
    const hipError_t launch_status = hipExtModuleLaunchKernel(
        function,
        global_work_items, 1, 1,
        local_work_size, 1, 1,
        0, stream, nullptr, reinterpret_cast<void**>(&config),
        nullptr, 0, 0);
    const hipError_t post_launch_status = hipGetLastError();
    TORCH_CHECK(launch_status == hipSuccess,
                "hipExtModuleLaunchKernel(K3 L2 asm) failed: ",
                hipGetErrorString(launch_status));
    TORCH_CHECK(post_launch_status == hipSuccess,
                "hipGetLastError after K3 L2 asm launch failed: ",
                hipGetErrorString(post_launch_status));
}

__global__ void build_row_combine_ptrs_kernel(
    uint8_t* local_sym_buffer,
    const int* output_index,
    const int* recv_to_global_token,
    uint64_t* row_combine_ptrs,
    const int total_rows,
    const int recv_rows,
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank,
    const int num_tokens,
    const int num_topk,
    const int hidden) {
    uint8_t** sym_buffers =
        deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);
    const int64_t routes = static_cast<int64_t>(recv_rows) * num_topk;
    for (int64_t route =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         route < routes;
         route += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = output_index[route];
        if (row < 0 || row >= total_rows)
            continue;
        const int recv_row = static_cast<int>(route / num_topk);
        const int topk_slot = static_cast<int>(route - static_cast<int64_t>(recv_row) * num_topk);
        const int global_token = recv_to_global_token[recv_row];
        if (global_token < 0)
            continue;
        const int source_rank = global_token / num_tokens;
        const int token_idx = global_token - source_rank * num_tokens;
        if (source_rank < 0 || source_rank >= num_ranks ||
            token_idx < 0 || token_idx >= num_tokens)
            continue;
        auto sections = deep_gemm::mega::get_sections(
            sym_buffers[source_rank], num_ranks, num_experts,
            num_max_tokens_per_rank, num_topk, hidden);
        const int64_t partial_row =
            static_cast<int64_t>(topk_slot) * num_max_tokens_per_rank + token_idx;
        row_combine_ptrs[row] =
            reinterpret_cast<uint64_t>(sections.combine + partial_row * hidden);
    }
}

__global__ void fill_row_combine_ptrs_kernel(
    uint64_t* row_combine_ptrs,
    const int total_rows) {
    for (int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < total_rows;
         row += static_cast<int>(gridDim.x) * blockDim.x) {
        row_combine_ptrs[row] = 0;
    }
}

__global__ void scatter_l2_to_combine_vec_kernel(
    const uint16_t* l2_out,
    const uint64_t* row_combine_ptrs,
    const int total_rows,
    const int hidden) {
    constexpr int kBf16PerVec = 8;
    const int vecs_per_row = hidden / kBf16PerVec;
    const int64_t total_vecs = static_cast<int64_t>(total_rows) * vecs_per_row;
    const auto* src_vec = reinterpret_cast<const uint4*>(l2_out);
    for (int64_t task =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         task < total_vecs;
         task += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(task / vecs_per_row);
        const int vec = static_cast<int>(task - static_cast<int64_t>(row) * vecs_per_row);
        const uint64_t ptr = row_combine_ptrs[row];
        if (ptr == 0)
            continue;
        auto* dst_vec = reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(ptr));
        dst_vec[vec] = src_vec[task];
    }
}

__global__ void reduce_local_combine_vec_kernel(
    uint16_t* y,
    uint8_t* local_sym_buffer,
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank,
    const int num_tokens,
    const int num_topk,
    const int hidden) {
    constexpr int kBf16PerVec = 8;
    auto local_sections = deep_gemm::mega::get_sections(
        local_sym_buffer, num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk, hidden);
    const int vecs_per_token = hidden / kBf16PerVec;
    const int64_t total_reduce_vecs =
        static_cast<int64_t>(num_tokens) * vecs_per_token;
    auto* y_vec = reinterpret_cast<uint4*>(y);
    for (int64_t task =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         task < total_reduce_vecs;
         task += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token_idx = static_cast<int>(task / vecs_per_token);
        const int vec_idx =
            static_cast<int>(task - static_cast<int64_t>(token_idx) * vecs_per_token);
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        float sum4 = 0.0f;
        float sum5 = 0.0f;
        float sum6 = 0.0f;
        float sum7 = 0.0f;
        for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {
            const int64_t partial_row =
                static_cast<int64_t>(topk_slot) * num_max_tokens_per_rank + token_idx;
            const auto packed =
                reinterpret_cast<const uint4*>(
                    local_sections.combine + partial_row * hidden)[vec_idx];
            sum0 += deep_gemm::mega::bf16_bits_to_float(static_cast<uint16_t>(packed.x));
            sum1 += deep_gemm::mega::bf16_bits_to_float(static_cast<uint16_t>(packed.x >> 16));
            sum2 += deep_gemm::mega::bf16_bits_to_float(static_cast<uint16_t>(packed.y));
            sum3 += deep_gemm::mega::bf16_bits_to_float(static_cast<uint16_t>(packed.y >> 16));
            sum4 += deep_gemm::mega::bf16_bits_to_float(static_cast<uint16_t>(packed.z));
            sum5 += deep_gemm::mega::bf16_bits_to_float(static_cast<uint16_t>(packed.z >> 16));
            sum6 += deep_gemm::mega::bf16_bits_to_float(static_cast<uint16_t>(packed.w));
            sum7 += deep_gemm::mega::bf16_bits_to_float(static_cast<uint16_t>(packed.w >> 16));
        }
        uint4 out;
        out.x = deep_gemm::mega::pack2_f32_to_bf16_bits(sum0, sum1);
        out.y = deep_gemm::mega::pack2_f32_to_bf16_bits(sum2, sum3);
        out.z = deep_gemm::mega::pack2_f32_to_bf16_bits(sum4, sum5);
        out.w = deep_gemm::mega::pack2_f32_to_bf16_bits(sum6, sum7);
        y_vec[task] = out;
    }
}

__global__ void rank_barrier_kernel(
    uint8_t* local_sym_buffer,
    const int rank_idx,
    const int num_ranks) {
    auto* signal_buffers =
        deep_gemm::mega::dcu_peer_signal_ptrs(local_sym_buffer, num_ranks);
    deep_gemm::mega::mega_moe_rank_barrier(signal_buffers, rank_idx, num_ranks);
}

__global__ void reset_asm_tail_signal_slots_kernel(
    uint8_t* local_sym_buffer,
    const int rank_idx,
    const int num_ranks) {
    auto* signal_buffers =
        deep_gemm::mega::dcu_peer_signal_ptrs(local_sym_buffer, num_ranks);
    auto* my_signals = signal_buffers[rank_idx];
    const int thread_id = static_cast<int>(threadIdx.x);
    if (thread_id < num_ranks) {
        __hip_atomic_store(my_signals + 8 + thread_id, 0, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

torch::Tensor k3_l2_original_asm(
    const torch::Tensor& act_fp8,
    const torch::Tensor& act_scale,
    const torch::Tensor& m_indices,
    const torch::Tensor& l2_weight,
    const torch::Tensor& l2_scale,
    const std::string& code_object_path) {
    const int64_t rows = act_fp8.size(0);
    const int64_t hidden = l2_weight.size(1) * 16;
    auto out = torch::empty(
        {rows, hidden},
        torch::TensorOptions().dtype(torch::kBFloat16).device(act_fp8.device()));
    launch_l2_deepgemm_original_asm(
        out, act_fp8, act_scale, m_indices, l2_weight, l2_scale,
        code_object_path, kAsmKernelName, nullptr);
    return out;
}

torch::Tensor k3_l2_combine_asm(
    const torch::Tensor& act_fp8,
    const torch::Tensor& act_scale,
    const torch::Tensor& m_indices,
    const torch::Tensor& l2_weight,
    const torch::Tensor& l2_scale,
    const torch::Tensor& row_combine_ptrs,
    const std::string& code_object_path) {
    const int64_t rows = act_fp8.size(0);
    const int64_t hidden = l2_weight.size(1) * 16;
    auto dummy_out = torch::empty(
        {rows, hidden},
        torch::TensorOptions().dtype(torch::kBFloat16).device(act_fp8.device()));
    launch_l2_deepgemm_original_asm(
        dummy_out, act_fp8, act_scale, m_indices, l2_weight, l2_scale,
        code_object_path, kCombineAsmKernelName, &row_combine_ptrs);
    return dummy_out;
}

torch::Tensor k3_l2_combine_asm_tail_reduce(
    const torch::Tensor& act_fp8,
    const torch::Tensor& act_scale,
    const torch::Tensor& m_indices,
    const torch::Tensor& l2_weight,
    const torch::Tensor& l2_scale,
    const torch::Tensor& row_combine_ptrs,
    const torch::Tensor& asm_done_counter,
    const torch::Tensor& asm_signal_addrs,
    const torch::Tensor& asm_reduce_y,
    const torch::Tensor& sym_buffer,
    const int64_t asm_done_target,
    const int64_t asm_signal_num_ranks,
    const int64_t asm_signal_generation,
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t num_topk,
    const int64_t hidden_arg,
    const std::string& code_object_path) {
    const int64_t rows = act_fp8.size(0);
    const int64_t hidden = l2_weight.size(1) * 16;
    TORCH_CHECK(hidden_arg == hidden,
                "hidden_arg must match L2 weight hidden");
    auto dummy_out = torch::empty(
        {rows, hidden},
        torch::TensorOptions().dtype(torch::kBFloat16).device(act_fp8.device()));
    launch_l2_deepgemm_original_asm(
        dummy_out, act_fp8, act_scale, m_indices, l2_weight, l2_scale,
        code_object_path, kCombineAsmKernelName, &row_combine_ptrs,
        &asm_done_counter, &asm_signal_addrs, asm_done_target,
        asm_signal_num_ranks, true, &asm_reduce_y, &sym_buffer,
        asm_signal_generation, num_ranks, num_experts,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden_arg);
    return dummy_out;
}

torch::Tensor build_row_combine_ptrs(
    const torch::Tensor& sym_buffer,
    const torch::Tensor& output_index,
    const torch::Tensor& recv_to_global_token,
    const int64_t total_rows,
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t num_topk,
    const int64_t hidden) {
    check_cuda_contiguous(sym_buffer, "sym_buffer");
    check_cuda_contiguous(output_index, "output_index");
    check_cuda_contiguous(recv_to_global_token, "recv_to_global_token");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be int8");
    TORCH_CHECK(output_index.scalar_type() == torch::kInt,
                "output_index must be int32");
    TORCH_CHECK(recv_to_global_token.scalar_type() == torch::kInt,
                "recv_to_global_token must be int32");
    TORCH_CHECK(output_index.dim() == 2,
                "output_index must be [recv_rows, topk]");
    TORCH_CHECK(output_index.size(1) == num_topk,
                "output_index topk dimension mismatch");
    TORCH_CHECK(recv_to_global_token.numel() == output_index.size(0),
                "recv_to_global_token rows mismatch");
    const int64_t row_ptr_count = total_rows + kAsmRowPointerPadding;
    auto row_ptrs = torch::empty(
        {row_ptr_count},
        torch::TensorOptions().dtype(torch::kInt64).device(sym_buffer.device()));
    constexpr int kThreads = 256;
    const int fill_blocks = std::max<int64_t>(
        1, std::min<int64_t>(4096, (row_ptr_count + kThreads - 1) / kThreads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(fill_row_combine_ptrs_kernel, dim3(fill_blocks), dim3(kThreads), 0, stream,
                       reinterpret_cast<uint64_t*>(row_ptrs.data_ptr<int64_t>()),
                       static_cast<int>(row_ptr_count));
    K3_HIP_CHECK(hipGetLastError());
    const int64_t routes = output_index.numel();
    const int blocks = std::max<int64_t>(1, std::min<int64_t>(4096, (routes + kThreads - 1) / kThreads));
    hipLaunchKernelGGL(build_row_combine_ptrs_kernel, dim3(blocks), dim3(kThreads), 0, stream,
                       reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
                       output_index.data_ptr<int>(),
                       recv_to_global_token.data_ptr<int>(),
                       reinterpret_cast<uint64_t*>(row_ptrs.data_ptr<int64_t>()),
                       static_cast<int>(total_rows),
                       static_cast<int>(output_index.size(0)),
                       static_cast<int>(num_ranks),
                       static_cast<int>(num_experts),
                       static_cast<int>(num_max_tokens_per_rank),
                       static_cast<int>(num_tokens),
                       static_cast<int>(num_topk),
                       static_cast<int>(hidden));
    K3_HIP_CHECK(hipGetLastError());
    return row_ptrs;
}

void scatter_l2_to_combine(
    const torch::Tensor& l2_out,
    const torch::Tensor& row_combine_ptrs) {
    check_cuda_contiguous(l2_out, "l2_out");
    check_cuda_contiguous(row_combine_ptrs, "row_combine_ptrs");
    TORCH_CHECK(l2_out.scalar_type() == torch::kBFloat16,
                "l2_out must be BF16");
    TORCH_CHECK(row_combine_ptrs.scalar_type() == torch::kInt64,
                "row_combine_ptrs must be int64");
    TORCH_CHECK(l2_out.dim() == 2, "l2_out must be [rows, hidden]");
    TORCH_CHECK(row_combine_ptrs.numel() >= l2_out.size(0),
                "row_combine_ptrs must cover every output row");
    const int total_rows = static_cast<int>(l2_out.size(0));
    const int hidden = static_cast<int>(l2_out.size(1));
    TORCH_CHECK(hidden % 8 == 0,
                "scatter_l2_to_combine currently expects hidden divisible by 8");
    constexpr int kThreads = 256;
    const int64_t total_vecs = static_cast<int64_t>(total_rows) * (hidden / 8);
    const int blocks = std::max<int64_t>(1, std::min<int64_t>(4096, (total_vecs + kThreads - 1) / kThreads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(scatter_l2_to_combine_vec_kernel, dim3(blocks), dim3(kThreads), 0, stream,
                       reinterpret_cast<const uint16_t*>(l2_out.data_ptr()),
                       reinterpret_cast<const uint64_t*>(row_combine_ptrs.data_ptr<int64_t>()),
                       total_rows,
                       hidden);
    K3_HIP_CHECK(hipGetLastError());
}

void reduce_local_combine(
    const torch::Tensor& y,
    const torch::Tensor& sym_buffer,
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t num_topk,
    const int64_t hidden) {
    check_cuda_contiguous(y, "y");
    check_cuda_contiguous(sym_buffer, "sym_buffer");
    TORCH_CHECK(y.scalar_type() == torch::kBFloat16, "y must be BF16");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be int8");
    TORCH_CHECK(y.dim() == 2 && y.size(0) == num_tokens && y.size(1) == hidden,
                "y shape mismatch");
    TORCH_CHECK(hidden % 8 == 0,
                "reduce_local_combine currently expects hidden divisible by 8");
    const int kThreads = std::max(64, std::min(256, env_int("K3_REDUCE_THREADS", 128)));
    const int64_t total_vecs = static_cast<int64_t>(num_tokens) * (hidden / 8);
    const int blocks = std::max<int64_t>(
        1, std::min<int64_t>(4096, (total_vecs + kThreads - 1) / kThreads));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(reduce_local_combine_vec_kernel, dim3(blocks), dim3(kThreads), 0, stream,
                       reinterpret_cast<uint16_t*>(y.data_ptr()),
                       reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
                       static_cast<int>(num_ranks),
                       static_cast<int>(num_experts),
                       static_cast<int>(num_max_tokens_per_rank),
                       static_cast<int>(num_tokens),
                       static_cast<int>(num_topk),
                       static_cast<int>(hidden));
    K3_HIP_CHECK(hipGetLastError());
}

void rank_barrier(
    const torch::Tensor& sym_buffer,
    const int64_t rank_idx,
    const int64_t num_ranks) {
    check_cuda_contiguous(sym_buffer, "sym_buffer");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be int8");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(rank_barrier_kernel, dim3(1), dim3(64), 0, stream,
                       reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
                       static_cast<int>(rank_idx),
                       static_cast<int>(num_ranks));
    K3_HIP_CHECK(hipGetLastError());
}

void reset_asm_tail_signal_slots(
    const torch::Tensor& sym_buffer,
    const int64_t rank_idx,
    const int64_t num_ranks) {
    check_cuda_contiguous(sym_buffer, "sym_buffer");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be int8");
    TORCH_CHECK(num_ranks > 0 && num_ranks <= 8,
                "reset_asm_tail_signal_slots supports num_ranks in [1, 8]");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(reset_asm_tail_signal_slots_kernel, dim3(1), dim3(64), 0, stream,
                       reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
                       static_cast<int>(rank_idx),
                       static_cast<int>(num_ranks));
    K3_HIP_CHECK(hipGetLastError());
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k3_l2_original_asm", &k3_l2_original_asm,
          pybind11::arg("act_fp8"),
          pybind11::arg("act_scale"),
          pybind11::arg("m_indices"),
          pybind11::arg("l2_weight"),
          pybind11::arg("l2_scale"),
          pybind11::arg("code_object_path"));
    m.def("k3_l2_combine_asm", &k3_l2_combine_asm,
          pybind11::arg("act_fp8"),
          pybind11::arg("act_scale"),
          pybind11::arg("m_indices"),
          pybind11::arg("l2_weight"),
          pybind11::arg("l2_scale"),
          pybind11::arg("row_combine_ptrs"),
          pybind11::arg("code_object_path"));
    m.def("k3_l2_combine_asm_tail_reduce", &k3_l2_combine_asm_tail_reduce,
          pybind11::arg("act_fp8"),
          pybind11::arg("act_scale"),
          pybind11::arg("m_indices"),
          pybind11::arg("l2_weight"),
          pybind11::arg("l2_scale"),
          pybind11::arg("row_combine_ptrs"),
          pybind11::arg("asm_done_counter"),
          pybind11::arg("asm_signal_addrs"),
          pybind11::arg("asm_reduce_y"),
          pybind11::arg("sym_buffer"),
          pybind11::arg("asm_done_target"),
          pybind11::arg("asm_signal_num_ranks"),
          pybind11::arg("asm_signal_generation"),
          pybind11::arg("num_ranks"),
          pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("hidden"),
          pybind11::arg("code_object_path"));
    m.def("build_row_combine_ptrs", &build_row_combine_ptrs,
          pybind11::arg("sym_buffer"),
          pybind11::arg("output_index"),
          pybind11::arg("recv_to_global_token"),
          pybind11::arg("total_rows"),
          pybind11::arg("num_ranks"),
          pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("hidden"));
    m.def("scatter_l2_to_combine", &scatter_l2_to_combine,
          pybind11::arg("l2_out"),
          pybind11::arg("row_combine_ptrs"));
    m.def("reduce_local_combine", &reduce_local_combine,
          pybind11::arg("y"),
          pybind11::arg("sym_buffer"),
          pybind11::arg("num_ranks"),
          pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("hidden"));
    m.def("rank_barrier", &rank_barrier,
          pybind11::arg("sym_buffer"),
          pybind11::arg("rank_idx"),
          pybind11::arg("num_ranks"));
    m.def("reset_asm_tail_signal_slots", &reset_asm_tail_signal_slots,
          pybind11::arg("sym_buffer"),
          pybind11::arg("rank_idx"),
          pybind11::arg("num_ranks"));
}
