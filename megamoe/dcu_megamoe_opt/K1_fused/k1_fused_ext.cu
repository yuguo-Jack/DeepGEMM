#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

#include <mega_moe_dcu/comm.cuh>
#include <mega_moe_dcu/layout.cuh>

void dcu_megamoe_v3_launch_k1_ll_symm_stage_pack5(
    hip_bfloat16* out,
    const uint8_t* staged_x,
    const uint8_t* weight_pack5,
    const float* staged_x_scale,
    const float* weight_scale,
    const int32_t* problem_size,
    uint8_t* sym_buffer,
    int32_t* route_scratch_i32,
    int32_t* grid_barrier,
    int barrier_epoch,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    int rows_aligned_per_expert,
    int valid_rows_per_expert,
    int ll_block_m,
    int ll_cus,
    float* route_weights,
    int32_t* row_expert_out,
    int32_t* output_index,
    int64_t* row_combine_ptrs,
    uint8_t* local_topk_mask,
    int32_t* tail_tokens,
    int32_t* cumulative_local_expert_recv_stats,
    bool enable_start_rank_barrier,
    int32_t* tail_done_counter,
    const int32_t* graph_runtime_num_tokens_for_barrier,
    int32_t* graph_runtime_num_tokens_out,
    int32_t* graph_tail_signal_generation_out,
    int graph_max_tokens,
    hipStream_t stream);

#define K1_HIP_CHECK(expr)                                                       \
    do {                                                                        \
        const hipError_t _status = (expr);                                       \
        TORCH_CHECK(_status == hipSuccess, #expr " failed: ",                  \
                    hipGetErrorString(_status));                                \
    } while (0)

namespace {

static constexpr const char* kFusedL1AsmKernelName =
    "DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1";
static constexpr int kK1SupportedRanks = 8;
static constexpr int kK1SupportedExperts = 256;
static constexpr int kK1SupportedTopk = 6;
static constexpr int kK1SupportedHidden = 4096;
static constexpr int kK1SupportedL1Rows = 4096;
static constexpr int kK1RouteTileM = 256;
static constexpr int kK1SupportedAlignment = 256;
static constexpr int kK1RouteCapacitySlack = 64;
static constexpr int kK1RouteCapacitySlackDivisor = 10;
static constexpr int64_t kK1RowPointerPadding = 512;
static constexpr double kK1AutoCompactMinSaving = 0.05;
static constexpr double kK1AutoCompactMinLocalTileSaving = 8.0;
static constexpr int64_t kK1AutoCompactHighTilesPerExpert = 7;
static constexpr double kK1CompactTightMarginMinSaving = 0.45;
static constexpr const char* kK1ShapeContract =
    "K1_fused dispatch-pull L1 asm is currently specialized for ranks=8, "
    "experts=256, local_experts=32, topk=6, hidden=4096, "
    "L1 output features=4096 (intermediate=2048), route_tile_m=256, "
    "alignment=256, and 0<=num_tokens_per_rank<=num_max_tokens_per_rank";

int64_t ceil_div_i64(const int64_t a, const int64_t b) {
    return (a + b - 1) / b;
}

int64_t route_capacity_headroom_rows(const int64_t expected_per_expert) {
    return std::max<int64_t>(
        kK1RouteCapacitySlack,
        ceil_div_i64(expected_per_expert, kK1RouteCapacitySlackDivisor));
}

double estimate_compact_tiles_per_expert(
    const int64_t total_tasks,
    const int64_t num_experts,
    const int64_t asm_tiles_per_expert) {
    if (total_tasks <= 0 || num_experts <= 0 || asm_tiles_per_expert <= 0) {
        return 0.0;
    }
    const double p = 1.0 / static_cast<double>(num_experts);
    const double mean = static_cast<double>(total_tasks) * p;
    const double variance = std::max(1.0, mean * (1.0 - p));
    const double sigma = std::sqrt(variance);
    const double inv_sqrt2 = 0.70710678118654752440;
    double tiles = 0.0;
    for (int64_t tile = 0; tile < asm_tiles_per_expert; ++tile) {
        const double threshold =
            static_cast<double>(tile * kK1RouteTileM);
        double prob = 1.0;
        if (tile == 0) {
            prob = 1.0 - std::exp(std::log1p(-p) *
                                  static_cast<double>(total_tasks));
        } else {
            const double z = (threshold + 0.5 - mean) / sigma;
            prob = 0.5 * std::erfc(z * inv_sqrt2);
        }
        tiles += std::clamp(prob, 0.0, 1.0);
    }
    return std::min<double>(tiles, static_cast<double>(asm_tiles_per_expert));
}

bool should_auto_compact_routes(
    const int64_t total_tasks,
    const int64_t num_experts,
    const int64_t local_experts,
    const int64_t asm_tiles_per_expert) {
    if (asm_tiles_per_expert <= 1) {
        return false;
    }
    if (asm_tiles_per_expert >= kK1AutoCompactHighTilesPerExpert) {
        return true;
    }
    const double compact_tiles =
        estimate_compact_tiles_per_expert(
            total_tasks, num_experts, asm_tiles_per_expert);
    const double saving =
        (static_cast<double>(asm_tiles_per_expert) - compact_tiles) /
        static_cast<double>(asm_tiles_per_expert);
    if (saving < kK1AutoCompactMinSaving) {
        return false;
    }
    const double local_tile_saving =
        (static_cast<double>(asm_tiles_per_expert) - compact_tiles) *
        static_cast<double>(local_experts);
    return local_tile_saving >= kK1AutoCompactMinLocalTileSaving;
}

int64_t compact_capacity_tiles(
    const int64_t total_tasks,
    const int64_t num_experts,
    const int64_t local_experts,
    const int64_t asm_tiles_per_expert) {
    const int64_t fixed_capacity_tiles = local_experts * asm_tiles_per_expert;
    if (asm_tiles_per_expert <= 1) {
        return fixed_capacity_tiles;
    }
    const double compact_tiles_per_expert =
        estimate_compact_tiles_per_expert(
            total_tasks, num_experts, asm_tiles_per_expert);
    const double estimated_tiles =
        compact_tiles_per_expert * static_cast<double>(local_experts);
    const double saving =
        (static_cast<double>(asm_tiles_per_expert) - compact_tiles_per_expert) /
        static_cast<double>(asm_tiles_per_expert);
    const int64_t margin_tiles =
        saving >= kK1CompactTightMarginMinSaving ? 2 : 4;
    int64_t capacity_tiles =
        static_cast<int64_t>(std::ceil(estimated_tiles)) +
        margin_tiles;
    capacity_tiles = std::max<int64_t>(local_experts, capacity_tiles);
    return std::min<int64_t>(fixed_capacity_tiles, capacity_tiles);
}

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
    void* staged_x;
    int32_t* staged_flags;
    void* symm_base;
    void* local_sym_buffer;
    void* route_scratch;
    uint32_t rank_idx;
    uint32_t num_ranks;
    uint32_t num_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_tokens;
    uint32_t num_topk;
    uint32_t reserved_c0;
    uint32_t reserved_c4;
    uint32_t flag_generation;
    uint32_t expert_tiles_per_expert;
    float* route_weights;
    int32_t* output_index;
    int64_t* row_combine_ptrs;
    int32_t* meta_flags;
    int32_t* local_expert_stats;
};

static_assert(offsetof(GpuProb, staged_x) == 0x80,
              "fused L1 asm expects staged_x at GpuProb+0x80");
static_assert(offsetof(GpuProb, staged_flags) == 0x88,
              "fused L1 asm expects staged_flags at GpuProb+0x88");
static_assert(offsetof(GpuProb, symm_base) == 0x90,
              "fused L1 asm expects symm_base at GpuProb+0x90");
static_assert(offsetof(GpuProb, local_sym_buffer) == 0x98,
              "route-in-asm path expects local_sym_buffer at GpuProb+0x98");
static_assert(offsetof(GpuProb, reserved_c0) == 0xc0,
              "fused L1 asm reserves GpuProb+0xc0");
static_assert(offsetof(GpuProb, flag_generation) == 0xc8,
              "fused L1 asm expects flag_generation at GpuProb+0xc8");
static_assert(offsetof(GpuProb, expert_tiles_per_expert) == 0xcc,
              "fused L1 asm expects expert_tiles_per_expert at GpuProb+0xcc");
static_assert(offsetof(GpuProb, route_weights) == 0xd0,
              "fused L1 asm expects route_weights at GpuProb+0xd0");
static_assert(offsetof(GpuProb, output_index) == 0xd8,
              "fused L1 asm expects output_index at GpuProb+0xd8");
static_assert(offsetof(GpuProb, row_combine_ptrs) == 0xe0,
              "fused L1 asm expects row_combine_ptrs at GpuProb+0xe0");
static_assert(offsetof(GpuProb, meta_flags) == 0xe8,
              "fused L1 asm expects meta_flags at GpuProb+0xe8");
static_assert(offsetof(GpuProb, local_expert_stats) == 0xf0,
              "fused L1 asm expects local_expert_stats at GpuProb+0xf0");

struct __attribute__((packed)) KernelArgs {
    uint32_t gemm_count;
    void const* DeviceUserArguments;
    void const* argsPtr;
    uint32_t kipWgTableGen;
    uint32_t gsu;
    int32_t* m_indics;
    int32_t* row_x_offsets;
    void* staged_x;
    int32_t* staged_flags;
    void* symm_base;
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

bool is_stream_capturing(const hipStream_t stream) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    const hipError_t err = hipStreamIsCapturing(stream, &status);
    return err == hipSuccess && status != hipStreamCaptureStatusNone;
}

uint32_t initial_fused_l1_flag_generation() {
    const auto now = std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count();
    uint64_t mixed = static_cast<uint64_t>(now);
    mixed ^= reinterpret_cast<uintptr_t>(&initial_fused_l1_flag_generation);
    mixed ^= mixed >> 32;
    uint32_t value = static_cast<uint32_t>(mixed);
    if (value == 0 || value < 0x10000u) {
        value ^= 0x4b1f0000u;
    }
    return value;
}

uint32_t next_fused_l1_flag_generation() {
    static std::atomic<uint32_t> generation{
        initial_fused_l1_flag_generation()};
    // The asm uses generation for init/reset flags and generation+1 for
    // meta-ready flags. Step by more than one so a previous ready flag cannot
    // masquerade as the next launch's init flag when route_scratch is reused.
    uint32_t value = generation.fetch_add(4, std::memory_order_relaxed) + 4;
    if (value == 0 || value == 0xffffffffu || value < 0x10000u) {
        value = generation.fetch_add(4, std::memory_order_relaxed) + 4;
    }
    return value;
}

__global__ void k1_init_compact_routes_kernel(
    int32_t* route_scratch_i32) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < 32; i += static_cast<int>(blockDim.x)) {
        route_scratch_i32[i] = 0;
    }
    if (tid == 0) {
        route_scratch_i32[64] = 0;
    }
}

__global__ void k1_count_compact_routes_kernel(
    uint8_t* local_sym_buffer,
    int32_t* route_scratch_i32,
    const int* runtime_num_tokens,
    const int rank_idx,
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank,
    const int num_tokens,
    const int num_topk,
    const int hidden) {
    const int tid = static_cast<int>(threadIdx.x);
    const int source_rank = static_cast<int>(blockIdx.y);
    int effective_num_tokens = num_tokens;
    const int local_experts = num_experts / num_ranks;
    const int first_expert = rank_idx * local_experts;
    const int last_expert = first_expert + local_experts;
    uint8_t** sym_buffers =
        deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);
    auto sections = deep_gemm::mega::get_sections(
        sym_buffers[source_rank], num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk, hidden);
    effective_num_tokens = sections.num_tokens[0];
    if (effective_num_tokens < 0) effective_num_tokens = 0;
    if (effective_num_tokens > num_max_tokens_per_rank) {
        effective_num_tokens = num_max_tokens_per_rank;
    }
    const int routes_per_rank = effective_num_tokens * num_topk;
    const int64_t* topk_idx = sections.topk_idx;
    for (int route_offset =
             static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
         route_offset < routes_per_rank;
         route_offset += static_cast<int>(gridDim.x) *
                         static_cast<int>(blockDim.x)) {
        const int64_t expert = topk_idx[route_offset];
        if (expert >= first_expert && expert < last_expert) {
            const int local_expert = static_cast<int>(expert) - first_expert;
            atomicAdd(&route_scratch_i32[local_expert], 1);
        }
    }
}

__global__ __launch_bounds__(1024) void k1_build_compact_tiles_kernel(
    int32_t* route_scratch_i32,
    float* route_weights,
    int64_t* row_x_ptrs,
    int64_t* row_combine_ptrs,
    float* row_x_scales,
    int32_t* m_indices,
    uint16_t* padding_combine_sink,
    const int capacity_tiles,
    const int runtime_limited_init,
    const int hidden) {
    constexpr int kLocalExperts = 32;
    constexpr int kTileM = kK1RouteTileM;
    __shared__ int active_tiles_shared;
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        int total_tiles = 0;
        for (int expert = 0; expert < kLocalExperts; ++expert) {
            route_scratch_i32[32 + expert] = total_tiles;
            int tiles = (route_scratch_i32[expert] + kTileM - 1) / kTileM;
            if (total_tiles + tiles > capacity_tiles) {
                tiles = capacity_tiles > total_tiles ? capacity_tiles - total_tiles : 0;
            }
            for (int tile = 0; tile < tiles; ++tile) {
                route_scratch_i32[65 + total_tiles + tile] = expert;
            }
            total_tiles += tiles;
            route_scratch_i32[expert] = 0;
        }
        route_scratch_i32[32 + kLocalExperts] = total_tiles;
        route_scratch_i32[64] = total_tiles;
        active_tiles_shared = total_tiles;
    }
    __syncthreads();
    const int active_rows = active_tiles_shared * kTileM;
    const int capacity_rows = capacity_tiles * kTileM;
    const int init_rows = runtime_limited_init != 0 ? active_rows : capacity_rows;
    for (int row = tid; row < init_rows; row += static_cast<int>(blockDim.x)) {
        const int tile_id = row / kTileM;
        const int expert =
            row < active_rows ? route_scratch_i32[65 + tile_id] : 0;
        row_x_ptrs[row] = -1;
        row_x_scales[row] = 0.0f;
        route_weights[row] = 0.0f;
        m_indices[row] = expert >= 0 ? expert : 0;
    }
    for (int row = tid;
         row < init_rows + static_cast<int>(kK1RowPointerPadding);
         row += static_cast<int>(blockDim.x)) {
        const int sink_row = row < init_rows ? row : 0;
        row_combine_ptrs[row] = padding_combine_sink == nullptr
            ? 0
            : static_cast<int64_t>(
                  reinterpret_cast<uintptr_t>(
                      padding_combine_sink +
                      static_cast<int64_t>(sink_row) * hidden));
    }
}

__global__ void k1_emit_compact_routes_kernel(
    uint8_t* local_sym_buffer,
    int32_t* route_scratch_i32,
    float* route_weights,
    int64_t* row_x_ptrs,
    int64_t* row_combine_ptrs,
    float* row_x_scales,
    int32_t* m_indices,
    int32_t* output_index,
    int32_t* local_expert_stats,
    const uint64_t symm_base_addr,
    const int* runtime_num_tokens,
    const int rank_idx,
    const int num_ranks,
    const int num_experts,
    const int num_max_tokens_per_rank,
    const int num_tokens,
    const int num_topk,
    const int hidden,
    const int capacity_tiles,
    const int use_rank_local_x_ptrs) {
    constexpr int kTileM = kK1RouteTileM;
    const int tid = static_cast<int>(threadIdx.x);
    const int source_rank = static_cast<int>(blockIdx.y);
    int effective_num_tokens = num_tokens;
    const int local_experts = num_experts / num_ranks;
    const int first_expert = rank_idx * local_experts;
    const int last_expert = first_expert + local_experts;
    const int active_tiles = route_scratch_i32[64];
    const int capacity_rows = capacity_tiles * kTileM;
    uint8_t** sym_buffers =
        deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);
    auto sections = deep_gemm::mega::get_sections(
        sym_buffers[source_rank], num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk, hidden);
    effective_num_tokens = sections.num_tokens[0];
    if (effective_num_tokens < 0) effective_num_tokens = 0;
    if (effective_num_tokens > num_max_tokens_per_rank) {
        effective_num_tokens = num_max_tokens_per_rank;
    }
    const int routes_per_rank = effective_num_tokens * num_topk;
    const int task_base = source_rank * num_max_tokens_per_rank * num_topk;
    const uint8_t* x = sections.x;
    const float* x_sf = sections.x_sf;
    const int64_t* topk_idx = sections.topk_idx;
    const float* topk_weights = sections.topk_weights;
    uint16_t* combine = sections.combine;
    for (int route_offset =
             static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
         route_offset < routes_per_rank;
         route_offset += static_cast<int>(gridDim.x) *
                         static_cast<int>(blockDim.x)) {
        const int task = task_base + route_offset;
        output_index[task] = -1;
        const int64_t expert = topk_idx[route_offset];
        if (expert < first_expert || expert >= last_expert) {
            continue;
        }
        const float weight = topk_weights[route_offset];
        if (weight == 0.0f) {
            continue;
        }
        const int local_expert = static_cast<int>(expert) - first_expert;
        const int row_in_expert =
            atomicAdd(&route_scratch_i32[local_expert], 1);
        const int tile_id =
            route_scratch_i32[32 + local_expert] + row_in_expert / kTileM;
        if (tile_id >= active_tiles || tile_id >= capacity_tiles) {
            continue;
        }
        const int row = tile_id * kTileM + row_in_expert % kTileM;
        if (row < 0 || row >= capacity_rows) {
            continue;
        }
        const int token_idx = route_offset / num_topk;
        const uint64_t x_addr =
            reinterpret_cast<uint64_t>(x) +
            static_cast<uint64_t>(token_idx) * static_cast<uint64_t>(hidden);
        output_index[task] = row;
        if (use_rank_local_x_ptrs) {
            const uint64_t peer_base_addr =
                reinterpret_cast<uint64_t>(sym_buffers[source_rank]);
            const uint64_t rank_local_offset = x_addr - peer_base_addr;
            row_x_ptrs[row] = static_cast<int64_t>(
                (static_cast<uint64_t>(source_rank) << 32) |
                static_cast<uint32_t>(rank_local_offset));
        } else {
            row_x_ptrs[row] =
                static_cast<int64_t>(
                    static_cast<uint32_t>(x_addr - symm_base_addr));
        }
        row_x_scales[row] = x_sf[token_idx];
        route_weights[row] = weight;
        m_indices[row] = local_expert;
        const int topk_slot = route_offset - token_idx * num_topk;
        const int64_t partial_row =
            static_cast<int64_t>(topk_slot) * num_max_tokens_per_rank + token_idx;
        row_combine_ptrs[row] =
            static_cast<int64_t>(
                reinterpret_cast<uintptr_t>(combine + partial_row * hidden));
        if (local_expert_stats != nullptr) {
            atomicAdd(local_expert_stats + local_expert, 1);
        }
    }
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
        K1_HIP_CHECK(hipModuleUnload(cache.module));
        cache.module = nullptr;
        cache.function = nullptr;
        cache.path.clear();
        cache.kernel_name.clear();
    }
    K1_HIP_CHECK(hipModuleLoad(&cache.module, code_object_path.c_str()));
    K1_HIP_CHECK(hipModuleGetFunction(&cache.function, cache.module, kernel_name));
    cache.path = code_object_path;
    cache.kernel_name = kernel_name;
    return cache.function;
}

void launch_l1_deepgemm_fused_asm(
    const torch::Tensor& output,
    const torch::Tensor& sym_buffer,
    const torch::Tensor& route_scratch,
    const uint64_t symm_base_addr,
    const torch::Tensor& row_x_ptrs,
    const torch::Tensor& row_x_scales,
    const torch::Tensor& staged_x,
    const torch::Tensor& l1_weight,
    const torch::Tensor& l1_scale,
    const torch::Tensor& m_indices,
    const torch::Tensor& route_weights,
    const torch::Tensor& row_combine_ptrs,
    const torch::Tensor& output_index,
    const torch::Tensor* local_expert_stats,
    const int64_t rank_idx,
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t route_capacity_num_tokens,
    const int64_t num_topk,
    const uint32_t expert_tiles_per_expert,
    const bool use_rank_local_x_ptrs,
    const bool use_compact_prebuild,
    const int* runtime_num_tokens,
    const std::string& code_object_path) {
    const int total_rows = static_cast<int>(row_x_ptrs.size(0));
    const int hidden = static_cast<int>(l1_weight.size(2) / 16);
    const int n = static_cast<int>(output.size(1));
    const int local_experts = static_cast<int>(l1_weight.size(0));
    TORCH_CHECK(hidden == kK1SupportedHidden && n == kK1SupportedL1Rows,
                kK1ShapeContract);
    TORCH_CHECK(num_ranks == kK1SupportedRanks &&
                    num_experts == kK1SupportedExperts &&
                    num_topk == kK1SupportedTopk,
                kK1ShapeContract);
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= num_max_tokens_per_rank,
                kK1ShapeContract);
    TORCH_CHECK(total_rows > 0 && total_rows % kK1RouteTileM == 0,
                "fused L1 asm expects total_rows padded to a 256-row tile");
    TORCH_CHECK(local_experts > 0, "local_experts must be positive");
    TORCH_CHECK(sym_buffer.is_cuda() && sym_buffer.scalar_type() == torch::kInt8 &&
                    sym_buffer.is_contiguous(),
                "sym_buffer must be contiguous CUDA int8");
    TORCH_CHECK(route_scratch.is_cuda() &&
                    route_scratch.scalar_type() == torch::kInt8 &&
                    route_scratch.is_contiguous(),
                "route_scratch must be contiguous CUDA int8");
    TORCH_CHECK(row_x_ptrs.is_cuda() && row_x_ptrs.scalar_type() == torch::kInt64 &&
                    row_x_ptrs.is_contiguous(),
                "row_x_ptrs must be a contiguous CUDA int64 tensor");
    TORCH_CHECK(row_x_scales.is_cuda() && row_x_scales.scalar_type() == torch::kFloat32 &&
                    row_x_scales.is_contiguous(),
                "row_x_scales must be a contiguous CUDA fp32 tensor");
    TORCH_CHECK(route_weights.is_cuda() && route_weights.scalar_type() == torch::kFloat32 &&
                    route_weights.is_contiguous() &&
                    route_weights.numel() >= total_rows,
                "route_weights must be a contiguous CUDA fp32 tensor");
    TORCH_CHECK(row_combine_ptrs.is_cuda() &&
                    row_combine_ptrs.scalar_type() == torch::kInt64 &&
                    row_combine_ptrs.is_contiguous() &&
                    row_combine_ptrs.numel() >= total_rows + kK1RowPointerPadding,
                "row_combine_ptrs must be a contiguous CUDA int64 tensor");
    TORCH_CHECK(output_index.is_cuda() && output_index.scalar_type() == torch::kInt &&
                    output_index.is_contiguous() &&
                    output_index.numel() >= num_ranks * num_tokens * num_topk,
                "output_index must be a contiguous CUDA int32 tensor");
    if (local_expert_stats != nullptr) {
        TORCH_CHECK(local_expert_stats->is_cuda() &&
                        local_expert_stats->scalar_type() == torch::kInt &&
                        local_expert_stats->is_contiguous() &&
                        local_expert_stats->numel() >= local_experts,
                    "local_expert_stats must be a contiguous CUDA int32 tensor");
    }
    const int wg_m = (n + 255) / 256;
    const int wg_n = (total_rows + 255) / 256;
    const int capacity_tiles = wg_n;
    TORCH_CHECK(staged_x.is_cuda() && staged_x.scalar_type() == torch::kFloat8_e4m3fn &&
                    staged_x.is_contiguous() &&
                    staged_x.numel() >= static_cast<int64_t>(total_rows) * hidden,
                "staged_x must be contiguous CUDA FP8 [total_rows, hidden]");
    const int64_t staged_tiles =
        (total_rows + kK1RouteTileM - 1) / kK1RouteTileM;
    const int64_t route_workspace_bytes =
        deep_gemm::mega::route_task_workspace_bytes(
            static_cast<int>(num_ranks),
            static_cast<int>(num_experts),
            static_cast<int>(num_max_tokens_per_rank));
    const int64_t staged_x_bytes =
        static_cast<int64_t>(total_rows) * hidden;
    const int64_t flags_offset =
        deep_gemm::mega::align_i64(route_workspace_bytes + staged_x_bytes, 16);
    const int64_t flags_bytes =
        staged_tiles * wg_m * static_cast<int64_t>(sizeof(int32_t));
    const int64_t meta_flags_offset =
        deep_gemm::mega::align_i64(flags_offset + flags_bytes, 16);
    const int64_t meta_flags_bytes =
        staged_tiles * static_cast<int64_t>(sizeof(int32_t));
    TORCH_CHECK(flags_offset >= 0 &&
                    meta_flags_offset + meta_flags_bytes <= route_scratch.numel(),
                "route_scratch is too small for K1 fused L1 flags");
    auto* staged_flags =
        reinterpret_cast<int32_t*>(
            static_cast<uint8_t*>(route_scratch.data_ptr()) + flags_offset);
    auto* meta_flags =
        reinterpret_cast<int32_t*>(
            static_cast<uint8_t*>(route_scratch.data_ptr()) + meta_flags_offset);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (use_compact_prebuild) {
        constexpr int route_threads = 256;
        const int routes_per_rank =
            static_cast<int>(route_capacity_num_tokens * num_topk);
        const int blocks_per_rank = runtime_num_tokens != nullptr
            ? 12
            : (routes_per_rank <= 8192 ? 6 : static_cast<int>(
                   std::min<int64_t>(
                       16,
                       std::max<int64_t>(
                           4, ceil_div_i64(routes_per_rank, 768)))));
        dim3 route_grid(blocks_per_rank, static_cast<unsigned>(num_ranks), 1);
        k1_init_compact_routes_kernel<<<1, route_threads, 0, stream>>>(
            reinterpret_cast<int32_t*>(route_scratch.data_ptr()));
        K1_HIP_CHECK(hipGetLastError());
        k1_count_compact_routes_kernel<<<route_grid, route_threads, 0, stream>>>(
            static_cast<uint8_t*>(sym_buffer.data_ptr()),
            reinterpret_cast<int32_t*>(route_scratch.data_ptr()),
            runtime_num_tokens,
            static_cast<int>(rank_idx),
            static_cast<int>(num_ranks),
            static_cast<int>(num_experts),
            static_cast<int>(num_max_tokens_per_rank),
            static_cast<int>(num_tokens),
            static_cast<int>(num_topk),
            hidden);
        K1_HIP_CHECK(hipGetLastError());
        k1_build_compact_tiles_kernel<<<1, 1024, 0, stream>>>(
            reinterpret_cast<int32_t*>(route_scratch.data_ptr()),
            route_weights.data_ptr<float>(),
            row_x_ptrs.data_ptr<int64_t>(),
            row_combine_ptrs.data_ptr<int64_t>(),
            row_x_scales.data_ptr<float>(),
            m_indices.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(output.data_ptr()),
            capacity_tiles,
            runtime_num_tokens == nullptr ? 0 : 1,
            hidden);
        K1_HIP_CHECK(hipGetLastError());
        k1_emit_compact_routes_kernel<<<route_grid, route_threads, 0, stream>>>(
            static_cast<uint8_t*>(sym_buffer.data_ptr()),
            reinterpret_cast<int32_t*>(route_scratch.data_ptr()),
            route_weights.data_ptr<float>(),
            row_x_ptrs.data_ptr<int64_t>(),
            row_combine_ptrs.data_ptr<int64_t>(),
            row_x_scales.data_ptr<float>(),
            m_indices.data_ptr<int32_t>(),
            output_index.data_ptr<int32_t>(),
            local_expert_stats == nullptr
                ? nullptr
                : local_expert_stats->data_ptr<int32_t>(),
            symm_base_addr,
            runtime_num_tokens,
            static_cast<int>(rank_idx),
            static_cast<int>(num_ranks),
            static_cast<int>(num_experts),
            static_cast<int>(num_max_tokens_per_rank),
            static_cast<int>(num_tokens),
            static_cast<int>(num_topk),
            hidden,
            capacity_tiles,
            use_rank_local_x_ptrs ? 1 : 0);
        K1_HIP_CHECK(hipGetLastError());
    }
    GpuProb prob{};
    prob.m = static_cast<uint32_t>(n);
    prob.n = static_cast<uint32_t>(total_rows);
    prob.batch = 1;
    prob.k = static_cast<uint32_t>(hidden);
    prob.d = output.data_ptr();
    prob.c = output.data_ptr();
    prob.a = l1_weight.data_ptr();
    prob.b = staged_x.data_ptr();
    prob.strideD1 = static_cast<uint32_t>(n);
    prob.strideD2 = static_cast<uint32_t>(total_rows * n);
    prob.strideC1 = static_cast<uint32_t>(n);
    prob.strideC2 = static_cast<uint32_t>(total_rows * n);
    prob.strideA1 = static_cast<uint32_t>(hidden);
    prob.strideA2 = static_cast<uint32_t>(n * hidden);
    prob.strideB1 = static_cast<uint32_t>(hidden);
    prob.strideB2 = static_cast<uint32_t>(total_rows * hidden);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    std::memcpy(prob.alpha, &alpha, sizeof(float));
    std::memcpy(prob.beta, &beta, sizeof(float));
    prob.scaleA = l1_scale.data_ptr<float>();
    prob.scaleB = row_x_scales.data_ptr<float>();
    prob.staged_x = staged_x.data_ptr();
    prob.staged_flags = staged_flags;
    prob.symm_base = reinterpret_cast<void*>(symm_base_addr);
    prob.local_sym_buffer = sym_buffer.data_ptr();
    prob.route_scratch = route_scratch.data_ptr();
    prob.rank_idx = static_cast<uint32_t>(rank_idx);
    prob.num_ranks = static_cast<uint32_t>(num_ranks);
    prob.num_experts = static_cast<uint32_t>(num_experts);
    prob.num_max_tokens_per_rank =
        static_cast<uint32_t>(num_max_tokens_per_rank);
    prob.num_tokens = static_cast<uint32_t>(num_tokens);
    prob.num_topk = static_cast<uint32_t>(num_topk);
    // reserved_c0 is a compact/asm-route mode bitfield:
    // bit0: metadata is prebuilt by HIP compact kernels.
    // bit2: row_x_ptrs stores {rank-local x offset, source rank}.
    // The bit2 path keeps source loads on MUBUF for >4GB spans, including
    // compact/graph prebuild.
    prob.reserved_c0 = (use_compact_prebuild ? 1u : 0u);
    if (use_rank_local_x_ptrs) {
        prob.reserved_c0 |= 4u;
    }
    prob.reserved_c4 = 0;
    prob.flag_generation = next_fused_l1_flag_generation();
    prob.expert_tiles_per_expert = expert_tiles_per_expert;
    prob.route_weights = route_weights.data_ptr<float>();
    prob.output_index = output_index.data_ptr<int32_t>();
    prob.row_combine_ptrs = row_combine_ptrs.data_ptr<int64_t>();
    prob.meta_flags = meta_flags;
    prob.local_expert_stats = local_expert_stats == nullptr
        ? nullptr
        : local_expert_stats->data_ptr<int32_t>();

    const int64_t prob_offset =
        deep_gemm::mega::align_i64(meta_flags_offset + meta_flags_bytes, 16);
    TORCH_CHECK(prob_offset + static_cast<int64_t>(sizeof(GpuProb)) <= route_scratch.numel(),
                "route_scratch is too small for K1 fused L1 launch arguments");
    void* prob_device = static_cast<uint8_t*>(route_scratch.data_ptr()) + prob_offset;
    if (!is_stream_capturing(stream)) {
        K1_HIP_CHECK(hipMemcpyAsync(
            prob_device, &prob, sizeof(GpuProb), hipMemcpyHostToDevice, stream));
    }
    KernelArgs args{};
    args.gemm_count = 1;
    args.DeviceUserArguments = prob_device;
    args.argsPtr = nullptr;
    args.kipWgTableGen = 0;
    args.gsu = 1;
    args.m_indics = m_indices.data_ptr<int32_t>();
    args.row_x_offsets =
        reinterpret_cast<int32_t*>(row_x_ptrs.data_ptr<int64_t>());
    args.staged_x = staged_x.data_ptr();
    args.staged_flags = staged_flags;
    args.symm_base = reinterpret_cast<void*>(symm_base_addr);

    const int local_work_size = 768;
    const size_t global_work_items =
        static_cast<size_t>(local_work_size) * wg_m * wg_n;

    hipFunction_t function =
        get_asm_function(code_object_path, kFusedL1AsmKernelName);
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
                "hipExtModuleLaunchKernel(fused L1 asm) failed: ",
                hipGetErrorString(launch_status));
    TORCH_CHECK(post_launch_status == hipSuccess,
                "hipGetLastError after fused L1 asm launch failed: ",
                hipGetErrorString(post_launch_status));
}

} // namespace

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
k1_symm_fused_l1_asm_impl(
    const torch::Tensor& sym_buffer,
    const torch::Tensor& route_scratch,
    const torch::Tensor& l1_weight,
    const torch::Tensor& l1_scale,
    const int64_t rank_idx,
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t num_topk,
    const int64_t hidden,
    const int64_t symm_base_addr_value,
    const int64_t symm_x_span_value,
    const int64_t alignment,
    const std::string& code_object_path,
    const std::optional<torch::Tensor>& l1_out_workspace,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor>& runtime_num_tokens,
    const bool force_compact_prebuild,
    const int64_t capacity_num_tokens) {
    TORCH_CHECK(sym_buffer.is_cuda() && route_scratch.is_cuda() &&
                    l1_weight.is_cuda() && l1_scale.is_cuda(),
                "symm K1 tensors must be CUDA/HIP tensors");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be int8");
    TORCH_CHECK(route_scratch.scalar_type() == torch::kInt8,
                "route_scratch must be int8");
    TORCH_CHECK(l1_weight.scalar_type() == torch::kFloat8_e4m3fn,
                "l1_weight must be FP8 E4M3");
    TORCH_CHECK(l1_scale.scalar_type() == torch::kFloat32,
                "l1_scale must be FP32");
    TORCH_CHECK(sym_buffer.is_contiguous() && route_scratch.is_contiguous() &&
                    l1_weight.is_contiguous() && l1_scale.is_contiguous(),
                "symm K1 tensors must be contiguous");
    TORCH_CHECK(!code_object_path.empty(),
                "K1_fused requires the fused L1 asm code object");
    TORCH_CHECK(num_ranks == kK1SupportedRanks &&
                    num_experts == kK1SupportedExperts &&
                    num_topk == kK1SupportedTopk &&
                    hidden == kK1SupportedHidden,
                kK1ShapeContract);
    TORCH_CHECK(rank_idx >= 0 && rank_idx < num_ranks, "invalid rank_idx");
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= num_max_tokens_per_rank,
                kK1ShapeContract);
    const int64_t route_capacity_num_tokens =
        capacity_num_tokens >= 0
            ? capacity_num_tokens
            : (force_compact_prebuild ? num_max_tokens_per_rank : num_tokens);
    TORCH_CHECK(route_capacity_num_tokens >= num_tokens &&
                    route_capacity_num_tokens <= num_max_tokens_per_rank,
                "capacity_num_tokens must be in [num_tokens, "
                "num_max_tokens_per_rank]");
    TORCH_CHECK(symm_base_addr_value > 0 && symm_x_span_value > 0,
                "invalid symm x address range");
    TORCH_CHECK(alignment == kK1SupportedAlignment, kK1ShapeContract);
    TORCH_CHECK(l1_scale.dim() == 2, "l1_scale must be [expert,row]");
    const int64_t local_experts = num_experts / num_ranks;
    const int64_t l1_rows = l1_scale.size(1);
    TORCH_CHECK(l1_rows == kK1SupportedL1Rows, kK1ShapeContract);
    TORCH_CHECK(l1_weight.size(0) == local_experts &&
                    l1_weight.size(1) == l1_rows / 16 &&
                    l1_weight.size(2) == hidden * 16,
                "invalid Marlin-packed L1 weight shape");

    const auto device = sym_buffer.device();
    const auto i32_options =
        torch::TensorOptions().dtype(torch::kInt).device(device);
    const auto f32_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto bf16_options =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    const auto fp8_options =
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);

    const int64_t total_tasks = num_ranks * num_tokens * num_topk;
    const int64_t max_stride_tasks =
        num_ranks * num_max_tokens_per_rank * num_topk;
    const int64_t capacity_total_tasks =
        num_ranks * route_capacity_num_tokens * num_topk;

    const int64_t expected_per_expert =
        (capacity_total_tasks + num_experts - 1) / num_experts;
    const int64_t rows_per_expert_target =
        std::max<int64_t>(
            alignment,
            expected_per_expert +
                route_capacity_headroom_rows(expected_per_expert));
    const int64_t fixed_capacity_tiles_per_expert =
        ceil_div_i64(rows_per_expert_target, kK1RouteTileM);
    bool use_compact_prebuild = force_compact_prebuild;
    bool force_asm_route = false;
    if (!force_compact_prebuild) {
        if (const char* mode = std::getenv("K1_PREBUILD_MODE")) {
            TORCH_CHECK(
                std::strcmp(mode, "auto") == 0 ||
                    std::strcmp(mode, "asm") == 0 ||
                    std::strcmp(mode, "asm_route") == 0 ||
                    std::strcmp(mode, "compact") == 0,
                "K1_PREBUILD_MODE supports only auto/asm/compact; fixed prebuild "
                "is intentionally disabled for K1_fused");
            use_compact_prebuild = std::strcmp(mode, "compact") == 0;
            force_asm_route =
                std::strcmp(mode, "asm") == 0 ||
                std::strcmp(mode, "asm_route") == 0;
        }
    }
    if (!use_compact_prebuild && !force_asm_route) {
        use_compact_prebuild = should_auto_compact_routes(
            capacity_total_tasks, num_experts, local_experts,
            fixed_capacity_tiles_per_expert);
    }
    const int64_t fixed_capacity_tiles =
        local_experts * fixed_capacity_tiles_per_expert;
    const int64_t capacity_tiles = use_compact_prebuild
                                       ? compact_capacity_tiles(
                                             capacity_total_tasks, num_experts,
                                             local_experts,
                                             fixed_capacity_tiles_per_expert)
                                       : fixed_capacity_tiles;
    const int64_t capacity_rows = capacity_tiles * kK1RouteTileM;
    const int64_t route_workspace_bytes =
        deep_gemm::mega::route_task_workspace_bytes(
            static_cast<int>(num_ranks),
            static_cast<int>(num_experts),
            static_cast<int>(num_max_tokens_per_rank));
    int64_t scratch_offset = 0;
    auto reserve_scratch = [&](const int64_t bytes) {
        const int64_t offset = deep_gemm::mega::align_i64(scratch_offset, 16);
        scratch_offset = offset + bytes;
        return offset;
    };
    // Route header:
    // counts[32], tile_bases[33], tile_experts[capacity_tiles].
    // The asm-route candidate additionally uses expert_tile_to_compact
    // [capacity_tiles] and per-row-tile stage flags [capacity_tiles * 16].
    reserve_scratch((local_experts + local_experts + 1 + capacity_tiles +
                     capacity_tiles + capacity_tiles * 16) *
                    static_cast<int64_t>(sizeof(int32_t)));
    const int64_t row_combine_ptrs_offset =
        reserve_scratch((capacity_rows + kK1RowPointerPadding) *
                        static_cast<int64_t>(sizeof(int64_t)));
    const int64_t route_weights_offset =
        reserve_scratch(capacity_rows * static_cast<int64_t>(sizeof(float)));
    const int64_t row_x_ptrs_offset =
        reserve_scratch(capacity_rows * static_cast<int64_t>(sizeof(int64_t)));
    const int64_t row_x_scales_offset =
        reserve_scratch(capacity_rows * static_cast<int64_t>(sizeof(float)));
    const int64_t m_indices_offset =
        reserve_scratch(capacity_rows * static_cast<int64_t>(sizeof(int32_t)));
    const int64_t output_index_tasks = max_stride_tasks;
    const int64_t output_index_offset =
        reserve_scratch(output_index_tasks * static_cast<int64_t>(sizeof(int32_t)));
    scratch_offset = deep_gemm::mega::align_i64(scratch_offset, 16);
    TORCH_CHECK(scratch_offset <= route_workspace_bytes,
                "route_scratch task workspace is too small for K1 metadata views");

    auto* scratch_base = static_cast<uint8_t*>(route_scratch.data_ptr());
    auto make_i32_view = [&](const int64_t offset, std::initializer_list<int64_t> shape) {
        return torch::from_blob(scratch_base + offset, shape, i32_options);
    };
    auto make_i64_view = [&](const int64_t offset, std::initializer_list<int64_t> shape) {
        return torch::from_blob(
            scratch_base + offset,
            shape,
            torch::TensorOptions().dtype(torch::kInt64).device(device));
    };
    auto make_f32_view = [&](const int64_t offset, std::initializer_list<int64_t> shape) {
        return torch::from_blob(scratch_base + offset, shape, f32_options);
    };
    auto row_combine_ptrs =
        make_i64_view(row_combine_ptrs_offset,
                      {capacity_rows + kK1RowPointerPadding});
    auto output_index =
        make_i32_view(output_index_offset, {output_index_tasks / num_topk, num_topk});
    const torch::Tensor* local_expert_stats = nullptr;
    if (cumulative_local_expert_recv_stats.has_value()) {
        const auto& stats = cumulative_local_expert_recv_stats.value();
        TORCH_CHECK(stats.is_cuda() && stats.is_contiguous() &&
                        stats.scalar_type() == torch::kInt &&
                        stats.numel() >= local_experts,
                    "cumulative_local_expert_recv_stats must be contiguous CUDA int32");
        local_expert_stats = &stats;
    }
    const int* runtime_num_tokens_ptr = nullptr;
    if (runtime_num_tokens.has_value()) {
        const auto& runtime = runtime_num_tokens.value();
        TORCH_CHECK(runtime.is_cuda() && runtime.is_contiguous() &&
                        runtime.scalar_type() == torch::kInt &&
                        runtime.numel() == 1,
                    "runtime_num_tokens must be a contiguous CUDA int32 scalar");
        runtime_num_tokens_ptr = runtime.data_ptr<int>();
    }
    const uint64_t symm_base_addr =
        static_cast<uint64_t>(symm_base_addr_value);
    bool use_rank_local_x_ptrs =
        static_cast<uint64_t>(symm_x_span_value) >
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    const int64_t total_rows = capacity_rows;
    const uint32_t expert_tiles_per_expert =
        static_cast<uint32_t>(fixed_capacity_tiles_per_expert);
    const int64_t staged_x_offset =
        deep_gemm::mega::align_i64(std::max(route_workspace_bytes, scratch_offset), 16);
    const int64_t staged_x_bytes = total_rows * static_cast<int64_t>(hidden);
    const int64_t flags_offset =
        deep_gemm::mega::align_i64(staged_x_offset + staged_x_bytes, 16);
    const int64_t flags_bytes =
        ((total_rows + 255) / 256) * ((l1_rows + 255) / 256) *
        static_cast<int64_t>(sizeof(int32_t));
    const int64_t meta_flags_offset =
        deep_gemm::mega::align_i64(flags_offset + flags_bytes, 16);
    const int64_t meta_flags_bytes =
        ((total_rows + 255) / 256) * static_cast<int64_t>(sizeof(int32_t));
    TORCH_CHECK(meta_flags_offset + meta_flags_bytes <= route_scratch.numel(),
                "route_scratch is too small for K1 staged_x and flags");

    auto route_weights = make_f32_view(route_weights_offset, {total_rows});
    auto row_x_ptrs = make_i64_view(row_x_ptrs_offset, {total_rows});
    auto row_x_scales = make_f32_view(row_x_scales_offset, {total_rows});
    auto m_indices = make_i32_view(m_indices_offset, {total_rows});
    auto staged_x =
        torch::from_blob(scratch_base + staged_x_offset,
                         {total_rows, hidden}, fp8_options);
    torch::Tensor l1_out;
    if (l1_out_workspace.has_value()) {
        const auto& workspace = l1_out_workspace.value();
        TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous(),
                    "l1_out_workspace must be contiguous CUDA/HIP memory");
        TORCH_CHECK(workspace.scalar_type() == torch::kBFloat16,
                    "l1_out_workspace must be BF16");
        TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) >= total_rows &&
                        workspace.size(1) == l1_rows,
                    "l1_out_workspace must be [>=total_rows, l1_rows]");
        l1_out = workspace.narrow(0, 0, total_rows);
    } else {
        l1_out = torch::empty({total_rows, l1_rows}, bf16_options);
    }
    launch_l1_deepgemm_fused_asm(
        l1_out, sym_buffer, route_scratch, symm_base_addr,
        row_x_ptrs, row_x_scales,
        staged_x, l1_weight, l1_scale, m_indices,
        route_weights, row_combine_ptrs, output_index,
        local_expert_stats,
        rank_idx, num_ranks, num_experts, num_max_tokens_per_rank,
        num_tokens, route_capacity_num_tokens, num_topk, expert_tiles_per_expert,
        use_rank_local_x_ptrs, use_compact_prebuild,
        runtime_num_tokens_ptr,
        code_object_path);

    return std::make_tuple(
        l1_out, route_weights, m_indices, output_index, row_combine_ptrs);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
k1_symm_fused_l1_v3_asm_pack5(
    const torch::Tensor& sym_buffer,
    const torch::Tensor& route_scratch,
    const torch::Tensor& l1_weight_pack5_asm,
    const torch::Tensor& l1_scale,
    const int64_t rank_idx,
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t num_topk,
    const int64_t hidden,
    const int64_t symm_base_addr_value,
    const int64_t symm_x_span_value,
    const int64_t alignment,
    const std::string& code_object_path,
    const std::optional<torch::Tensor>& l1_out_workspace,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor>& runtime_num_tokens,
    const bool force_compact_prebuild,
    const int64_t capacity_num_tokens) {
    return k1_symm_fused_l1_asm_impl(
        sym_buffer, route_scratch, l1_weight_pack5_asm, l1_scale,
        rank_idx, num_ranks, num_experts, num_max_tokens_per_rank,
        num_tokens, num_topk, hidden, symm_base_addr_value,
        symm_x_span_value, alignment, code_object_path,
        l1_out_workspace, cumulative_local_expert_recv_stats,
        runtime_num_tokens, force_compact_prebuild, capacity_num_tokens);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
k1_symm_fused_l1_v3_pack5(
    const torch::Tensor& sym_buffer,
    const torch::Tensor& route_scratch,
    const torch::Tensor& l1_weight_pack5,
    const torch::Tensor& l1_scale,
    const int64_t rank_idx,
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t num_topk,
    const int64_t hidden,
    const int64_t alignment,
    const std::string& backend,
    const std::optional<torch::Tensor>& l1_out_workspace,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const std::optional<torch::Tensor>& runtime_num_tokens,
    const int64_t ll_block_m,
    const int64_t ll_cus,
    const bool ll_asm_compatible_layout,
    const int64_t capacity_num_tokens,
    const bool enable_start_rank_barrier,
    const std::optional<torch::Tensor>& tail_done_counter,
    const std::optional<torch::Tensor>& graph_runtime_num_tokens_out,
    const std::optional<torch::Tensor>& graph_tail_signal_generation_out) {
    TORCH_CHECK(sym_buffer.is_cuda() && route_scratch.is_cuda() &&
                    l1_weight_pack5.is_cuda() && l1_scale.is_cuda(),
                "V3 K1 tensors must be CUDA/HIP tensors");
    TORCH_CHECK(sym_buffer.scalar_type() == torch::kInt8,
                "sym_buffer must be int8");
    TORCH_CHECK(route_scratch.scalar_type() == torch::kInt8,
                "route_scratch must be int8");
    TORCH_CHECK(l1_weight_pack5.scalar_type() == torch::kFloat8_e4m3fn,
                "V3 K1 l1_weight_pack5 must be FP8 E4M3");
    TORCH_CHECK(l1_scale.scalar_type() == torch::kFloat32,
                "V3 K1 l1_scale must be FP32");
    TORCH_CHECK(sym_buffer.is_contiguous() && route_scratch.is_contiguous() &&
                    l1_weight_pack5.is_contiguous() && l1_scale.is_contiguous(),
                "V3 K1 tensors must be contiguous");
    TORCH_CHECK(num_ranks == kK1SupportedRanks &&
                    num_experts == kK1SupportedExperts &&
                    num_topk == kK1SupportedTopk &&
                    hidden == kK1SupportedHidden &&
                    alignment == kK1SupportedAlignment,
                kK1ShapeContract);
    TORCH_CHECK(rank_idx >= 0 && rank_idx < num_ranks, "invalid rank_idx");
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= num_max_tokens_per_rank,
                kK1ShapeContract);
    TORCH_CHECK(backend == "ll",
                "V3 K1 pack5 C entry is LL-only; normal uses ASM-pack5");

    const int64_t local_experts = num_experts / num_ranks;
    TORCH_CHECK(local_experts == 32, kK1ShapeContract);
    TORCH_CHECK(l1_scale.dim() == 2 &&
                    l1_scale.size(0) == local_experts &&
                    l1_scale.size(1) == kK1SupportedL1Rows,
                "V3 K1 l1_scale must be [local_experts, 4096]");
    TORCH_CHECK(l1_weight_pack5.dim() >= 2 &&
                    l1_weight_pack5.size(0) == local_experts &&
                    l1_weight_pack5.numel() >=
                        local_experts * static_cast<int64_t>(kK1SupportedL1Rows) *
                            kK1SupportedHidden,
                "V3 K1 l1_weight_pack5 must cover [local_experts, 4096 * 4096]");
    if (runtime_num_tokens.has_value()) {
        const auto& runtime = runtime_num_tokens.value();
        TORCH_CHECK(runtime.is_cuda() && runtime.is_contiguous() &&
                        runtime.scalar_type() == torch::kInt &&
                        runtime.numel() == 1,
                    "runtime_num_tokens must be a contiguous CUDA int32 scalar");
    }
    const torch::Tensor* tail_done_counter_tensor = nullptr;
    if (tail_done_counter.has_value()) {
        const auto& counter = tail_done_counter.value();
        TORCH_CHECK(counter.is_cuda() && counter.is_contiguous() &&
                        counter.scalar_type() == torch::kInt &&
                        counter.numel() >= 80,
                    "tail_done_counter must be a contiguous CUDA int32 tensor "
                    "with at least 80 elements");
        tail_done_counter_tensor = &counter;
    }
    const torch::Tensor* graph_runtime_num_tokens_out_tensor = nullptr;
    if (graph_runtime_num_tokens_out.has_value()) {
        const auto& runtime_out = graph_runtime_num_tokens_out.value();
        TORCH_CHECK(runtime_out.is_cuda() && runtime_out.is_contiguous() &&
                        runtime_out.scalar_type() == torch::kInt &&
                        runtime_out.numel() >= 1,
                    "graph_runtime_num_tokens_out must be a contiguous CUDA "
                    "int32 tensor");
        graph_runtime_num_tokens_out_tensor = &runtime_out;
    }
    const torch::Tensor* graph_tail_signal_generation_out_tensor = nullptr;
    if (graph_tail_signal_generation_out.has_value()) {
        const auto& generation_out = graph_tail_signal_generation_out.value();
        TORCH_CHECK(generation_out.is_cuda() && generation_out.is_contiguous() &&
                        generation_out.scalar_type() == torch::kInt &&
                        generation_out.numel() >= 1,
                    "graph_tail_signal_generation_out must be a contiguous CUDA "
                    "int32 tensor");
        graph_tail_signal_generation_out_tensor = &generation_out;
    }
    TORCH_CHECK(!enable_start_rank_barrier || tail_done_counter_tensor != nullptr,
                "LL start rank barrier requires tail_done_counter");
    TORCH_CHECK(!graph_runtime_num_tokens_out.has_value() ||
                    runtime_num_tokens.has_value(),
                "graph_runtime_num_tokens_out requires runtime_num_tokens");

    const bool use_ll = true;
    const int64_t ll_row_tile =
        ll_asm_compatible_layout ? kK1RouteTileM : 64;
    const int64_t row_tile = use_ll ? ll_row_tile : kK1RouteTileM;
    const int64_t route_capacity_tokens_per_rank =
        capacity_num_tokens >= 0 ? capacity_num_tokens : num_tokens;
    TORCH_CHECK(route_capacity_tokens_per_rank >= num_tokens &&
                    route_capacity_tokens_per_rank <= num_max_tokens_per_rank,
                "capacity_num_tokens must be in [num_tokens, "
                "num_max_tokens_per_rank]");
    const int64_t capacity_total_tasks =
        num_ranks * route_capacity_tokens_per_rank * num_topk;
    const int64_t expected_per_expert =
        (capacity_total_tasks + num_experts - 1) / num_experts;
    const int64_t rows_per_expert_target =
        std::max<int64_t>(
            alignment,
            expected_per_expert +
                route_capacity_headroom_rows(expected_per_expert));
    const int64_t fixed_capacity_tiles_per_expert =
        ceil_div_i64(rows_per_expert_target, kK1RouteTileM);
    const int64_t fixed_capacity_tiles =
        local_experts * fixed_capacity_tiles_per_expert;
    const int64_t compact_capacity_tiles_i64 =
        compact_capacity_tiles(
            capacity_total_tasks, num_experts, local_experts,
            fixed_capacity_tiles_per_expert);
    const bool use_compact_capacity =
        !use_ll && compact_capacity_tiles_i64 < fixed_capacity_tiles;
    const int64_t ll_base_rows_per_expert =
        deep_gemm::mega::align_i64(
            std::max<int64_t>(
                1,
                ceil_div_i64(
                    route_capacity_tokens_per_rank * num_topk,
                    local_experts)),
            row_tile);
    const int64_t ll_expected_rows_per_expert =
        std::max<int64_t>(
            1,
            ceil_div_i64(
                route_capacity_tokens_per_rank * num_topk,
                local_experts));
    constexpr int64_t kLlHeadroomExpectedRowsThreshold = 48;
    constexpr int64_t kLlHighHeadroomExpectedRowsThreshold = 512;
    constexpr int64_t kLlHeadroomRows = 64;
    constexpr int64_t kLlHighHeadroomRows = 128;
    // Preserve tiny LL buckets exactly, but reserve enough per-expert headroom
    // for high-token buckets: random routing can exceed the mean per-expert count.
    const int64_t ll_min_slack =
        ll_expected_rows_per_expert >= kLlHighHeadroomExpectedRowsThreshold
            ? kLlHighHeadroomRows
            : (ll_expected_rows_per_expert >= kLlHeadroomExpectedRowsThreshold
                   ? kLlHeadroomRows
                   : 0);
    const int64_t ll_rows_per_expert =
        ll_base_rows_per_expert - ll_expected_rows_per_expert < ll_min_slack
            ? deep_gemm::mega::align_i64(
                  ll_expected_rows_per_expert + ll_min_slack,
                  row_tile)
            : ll_base_rows_per_expert;
    const int64_t capacity_tiles_i64 =
        use_ll
            ? local_experts * ceil_div_i64(ll_rows_per_expert, kK1RouteTileM)
            : (use_compact_capacity
                   ? compact_capacity_tiles_i64
                   : fixed_capacity_tiles);
    const int64_t normal_total_rows = capacity_tiles_i64 * kK1RouteTileM;
    const int64_t rows_aligned_per_expert =
        use_ll ? ll_rows_per_expert
               : normal_total_rows;
    const int64_t valid_rows_per_expert =
        use_ll ? ll_expected_rows_per_expert
               : normal_total_rows;
    const int64_t ll_launch_rows = local_experts * rows_aligned_per_expert;
    const int64_t total_rows =
        use_ll ? ll_launch_rows : normal_total_rows;
    const int64_t output_index_tasks =
        num_ranks * num_max_tokens_per_rank * num_topk;
    const int64_t route_workspace_bytes =
        deep_gemm::mega::route_task_workspace_bytes(
            static_cast<int>(num_ranks),
            static_cast<int>(num_experts),
            static_cast<int>(num_max_tokens_per_rank));

    int64_t scratch_offset = 0;
    auto reserve_scratch = [&](const int64_t bytes) {
        const int64_t offset = deep_gemm::mega::align_i64(scratch_offset, 16);
        scratch_offset = offset + bytes;
        return offset;
    };
    const int64_t route_scratch_i32_ints =
        use_ll ? (local_experts + 2 +
                  2 * local_experts * rows_aligned_per_expert)
               : (local_experts + local_experts + 1 + capacity_tiles_i64 +
                  capacity_tiles_i64 + capacity_tiles_i64 * 16);
    const int64_t grid_barrier_ints =
        use_ll ? 2 : (16 * ceil_div_i64(total_rows, kK1RouteTileM) + 2);
    const int64_t route_scratch_i32_offset =
        reserve_scratch(route_scratch_i32_ints * static_cast<int64_t>(sizeof(int32_t)));
    const int64_t grid_barrier_offset =
        reserve_scratch(grid_barrier_ints * static_cast<int64_t>(sizeof(int32_t)));
    const int64_t local_topk_mask_offset =
        reserve_scratch(num_max_tokens_per_rank * static_cast<int64_t>(sizeof(uint8_t)));
    const int64_t tail_tokens_offset =
        reserve_scratch(num_max_tokens_per_rank * static_cast<int64_t>(sizeof(int32_t)));
    const int64_t row_combine_ptrs_offset =
        reserve_scratch((total_rows + kK1RowPointerPadding) *
                        static_cast<int64_t>(sizeof(int64_t)));
    const int64_t route_weights_offset =
        reserve_scratch(total_rows * static_cast<int64_t>(sizeof(float)));
    const int64_t row_x_ptrs_offset =
        reserve_scratch(total_rows * static_cast<int64_t>(sizeof(int64_t)));
    const int64_t row_x_scales_offset =
        reserve_scratch(total_rows * static_cast<int64_t>(sizeof(float)));
    const int64_t m_indices_offset =
        reserve_scratch(total_rows * static_cast<int64_t>(sizeof(int32_t)));
    const int64_t output_index_offset =
        reserve_scratch(output_index_tasks * static_cast<int64_t>(sizeof(int32_t)));
    const int64_t staged_x_scale_offset =
        reserve_scratch(total_rows * static_cast<int64_t>(sizeof(float)));
    scratch_offset = deep_gemm::mega::align_i64(scratch_offset, 16);
    TORCH_CHECK(scratch_offset <= route_workspace_bytes,
                "route_scratch task workspace is too small for V3 K1 metadata");
    const int64_t staged_x_offset = deep_gemm::mega::align_i64(route_workspace_bytes, 16);
    const int64_t staged_x_bytes = total_rows * hidden;
    TORCH_CHECK(staged_x_offset + staged_x_bytes <= route_scratch.numel(),
                "route_scratch is too small for V3 K1 staged_x workspace");

    const auto device = sym_buffer.device();
    const auto i32_options =
        torch::TensorOptions().dtype(torch::kInt).device(device);
    const auto i64_options =
        torch::TensorOptions().dtype(torch::kInt64).device(device);
    const auto f32_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto u8_options =
        torch::TensorOptions().dtype(torch::kUInt8).device(device);
    const auto bf16_options =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    const auto fp8_options =
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);
    auto* scratch_base = static_cast<uint8_t*>(route_scratch.data_ptr());
    auto make_i32_view = [&](const int64_t offset, std::initializer_list<int64_t> shape) {
        return torch::from_blob(scratch_base + offset, shape, i32_options);
    };
    auto make_i64_view = [&](const int64_t offset, std::initializer_list<int64_t> shape) {
        return torch::from_blob(scratch_base + offset, shape, i64_options);
    };
    auto make_f32_view = [&](const int64_t offset, std::initializer_list<int64_t> shape) {
        return torch::from_blob(scratch_base + offset, shape, f32_options);
    };
    auto route_scratch_i32 =
        make_i32_view(route_scratch_i32_offset, {route_scratch_i32_ints});
    // K1 LL writes per-local-expert row counts at the head of route_scratch_i32.
    // The extra slot carries max(actual_m) for K2; K3 only consumes the first
    // local_experts entries.
    auto ll_actual_m =
        make_i32_view(route_scratch_i32_offset, {local_experts + 1});
    auto grid_barrier = make_i32_view(grid_barrier_offset, {grid_barrier_ints});
    auto local_topk_mask =
        torch::from_blob(
            scratch_base + local_topk_mask_offset,
            {num_max_tokens_per_rank},
            u8_options);
    auto tail_tokens = make_i32_view(tail_tokens_offset, {num_max_tokens_per_rank});
    auto row_combine_ptrs =
        make_i64_view(row_combine_ptrs_offset,
                      {total_rows + kK1RowPointerPadding});
    auto route_weights = make_f32_view(route_weights_offset, {total_rows});
    auto row_x_ptrs = make_i64_view(row_x_ptrs_offset, {total_rows});
    auto row_x_scales = make_f32_view(row_x_scales_offset, {total_rows});
    auto m_indices = make_i32_view(m_indices_offset, {total_rows});
    auto output_index =
        make_i32_view(output_index_offset,
                      {output_index_tasks / num_topk, num_topk});
    auto staged_x_scale = make_f32_view(staged_x_scale_offset, {total_rows});
    auto staged_x =
        torch::from_blob(scratch_base + staged_x_offset,
                         {total_rows, hidden}, fp8_options);

    torch::Tensor l1_out;
    if (l1_out_workspace.has_value()) {
        const auto& workspace = l1_out_workspace.value();
        TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous(),
                    "V3 K1 l1_out_workspace must be contiguous CUDA/HIP memory");
        TORCH_CHECK(workspace.scalar_type() == torch::kBFloat16,
                    "V3 K1 l1_out_workspace must be BF16");
        TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) >= total_rows &&
                        workspace.size(1) == kK1SupportedL1Rows,
                    "V3 K1 l1_out_workspace must be [>=total_rows, 4096]");
        l1_out = workspace.narrow(0, 0, total_rows);
    } else {
        l1_out = torch::empty({total_rows, kK1SupportedL1Rows}, bf16_options);
    }
    const torch::Tensor* local_expert_stats = nullptr;
    if (cumulative_local_expert_recv_stats.has_value()) {
        const auto& stats = cumulative_local_expert_recv_stats.value();
        TORCH_CHECK(stats.is_cuda() && stats.is_contiguous() &&
                        stats.scalar_type() == torch::kInt &&
                        stats.numel() >= local_experts,
                    "cumulative_local_expert_recv_stats must be contiguous CUDA int32");
        local_expert_stats = &stats;
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int runtime_launch_num_tokens =
        runtime_num_tokens.has_value() ? -1 : static_cast<int>(num_tokens);
    TORCH_CHECK((ll_block_m == 32 || ll_block_m == 48 || ll_block_m == 64) &&
                    ll_cus == 64,
                "V3 K1 ll supports ll_block_m in {32,48,64} with ll_cus=64");
    const int epoch = static_cast<int>(next_fused_l1_flag_generation());
    dcu_megamoe_v3_launch_k1_ll_symm_stage_pack5(
        reinterpret_cast<hip_bfloat16*>(l1_out.data_ptr()),
        reinterpret_cast<const uint8_t*>(staged_x.data_ptr()),
        reinterpret_cast<const uint8_t*>(l1_weight_pack5.data_ptr()),
        staged_x_scale.data_ptr<float>(),
        l1_scale.data_ptr<float>(),
        route_scratch_i32.data_ptr<int32_t>(),
        reinterpret_cast<uint8_t*>(sym_buffer.data_ptr<int8_t>()),
        route_scratch_i32.data_ptr<int32_t>(),
        grid_barrier.data_ptr<int32_t>(),
        epoch,
        static_cast<int>(rank_idx),
        static_cast<int>(num_ranks),
        static_cast<int>(num_experts),
        static_cast<int>(num_max_tokens_per_rank),
        static_cast<int>(num_topk),
        runtime_launch_num_tokens,
        static_cast<int>(rows_aligned_per_expert),
        static_cast<int>(valid_rows_per_expert),
        static_cast<int>(ll_block_m),
        static_cast<int>(ll_cus),
        route_weights.data_ptr<float>(),
        m_indices.data_ptr<int32_t>(),
        output_index.data_ptr<int32_t>(),
        row_combine_ptrs.data_ptr<int64_t>(),
        nullptr,
        nullptr,
        local_expert_stats == nullptr
            ? nullptr
            : local_expert_stats->data_ptr<int32_t>(),
        enable_start_rank_barrier,
        tail_done_counter_tensor == nullptr
            ? nullptr
            : tail_done_counter_tensor->data_ptr<int32_t>(),
        runtime_num_tokens.has_value()
            ? runtime_num_tokens.value().data_ptr<int32_t>()
            : nullptr,
        graph_runtime_num_tokens_out_tensor == nullptr
            ? nullptr
            : graph_runtime_num_tokens_out_tensor->data_ptr<int32_t>(),
        graph_tail_signal_generation_out_tensor == nullptr
            ? nullptr
            : graph_tail_signal_generation_out_tensor->data_ptr<int32_t>(),
        enable_start_rank_barrier ? static_cast<int>(num_tokens) : -1,
        stream);
    K1_HIP_CHECK(hipGetLastError());

    return std::make_tuple(
        l1_out, route_weights, ll_actual_m, output_index, row_combine_ptrs);
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
k1_graph_flag_reset_layout(
    const int64_t num_ranks,
    const int64_t num_experts,
    const int64_t num_max_tokens_per_rank,
    const int64_t num_tokens,
    const int64_t num_topk,
    const int64_t hidden,
    const int64_t l1_rows,
    const int64_t alignment) {
    TORCH_CHECK(num_ranks == kK1SupportedRanks &&
                    num_experts == kK1SupportedExperts &&
                    num_topk == kK1SupportedTopk &&
                    hidden == kK1SupportedHidden &&
                    l1_rows == kK1SupportedL1Rows &&
                    alignment == kK1SupportedAlignment &&
                    num_tokens >= 0 && num_tokens <= num_max_tokens_per_rank,
                kK1ShapeContract);
    const int64_t local_experts = num_experts / num_ranks;
    const int64_t total_tasks = num_ranks * num_max_tokens_per_rank * num_topk;
    const int64_t expected_per_expert =
        (total_tasks + num_experts - 1) / num_experts;
    const int64_t rows_per_expert_target =
        std::max<int64_t>(
            alignment,
            expected_per_expert +
                route_capacity_headroom_rows(expected_per_expert));
    const int64_t fixed_capacity_tiles_per_expert =
        ceil_div_i64(rows_per_expert_target, kK1RouteTileM);
    const int64_t fixed_capacity_tiles =
        local_experts * fixed_capacity_tiles_per_expert;
    const int64_t capacity_tiles = compact_capacity_tiles(
        total_tasks, num_experts, local_experts, fixed_capacity_tiles_per_expert);
    const int64_t total_rows = capacity_tiles * kK1RouteTileM;
    const int64_t route_workspace_bytes =
        deep_gemm::mega::route_task_workspace_bytes(
            static_cast<int>(num_ranks),
            static_cast<int>(num_experts),
            static_cast<int>(num_max_tokens_per_rank));
    int64_t scratch_offset = 0;
    auto reserve_scratch = [&](const int64_t bytes) {
        const int64_t offset = deep_gemm::mega::align_i64(scratch_offset, 16);
        scratch_offset = offset + bytes;
        return offset;
    };
    reserve_scratch((local_experts + local_experts + 1 + capacity_tiles +
                     capacity_tiles + capacity_tiles * 16) *
                    static_cast<int64_t>(sizeof(int32_t)));
    reserve_scratch((total_rows + kK1RowPointerPadding) *
                    static_cast<int64_t>(sizeof(int64_t)));
    reserve_scratch(total_rows * static_cast<int64_t>(sizeof(float)));
    reserve_scratch(total_rows * static_cast<int64_t>(sizeof(int64_t)));
    reserve_scratch(total_rows * static_cast<int64_t>(sizeof(float)));
    reserve_scratch(total_rows * static_cast<int64_t>(sizeof(int32_t)));
    reserve_scratch(total_tasks * static_cast<int64_t>(sizeof(int32_t)));
    scratch_offset = deep_gemm::mega::align_i64(scratch_offset, 16);
    const int64_t staged_x_offset =
        deep_gemm::mega::align_i64(std::max(route_workspace_bytes, scratch_offset), 16);
    const int64_t staged_x_bytes = total_rows * hidden;
    const int64_t flags_offset =
        deep_gemm::mega::align_i64(staged_x_offset + staged_x_bytes, 16);
    const int64_t flags_numel =
        ((total_rows + 255) / 256) * ((l1_rows + 255) / 256);
    const int64_t meta_flags_offset =
        deep_gemm::mega::align_i64(flags_offset + flags_numel * sizeof(int32_t), 16);
    const int64_t meta_flags_numel = (total_rows + 255) / 256;
    return std::make_tuple(flags_offset, flags_numel, meta_flags_offset,
                           meta_flags_numel, total_rows, fixed_capacity_tiles);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("k1_symm_fused_l1_v3_asm_pack5", &k1_symm_fused_l1_v3_asm_pack5,
          pybind11::arg("sym_buffer"),
          pybind11::arg("route_scratch"),
          pybind11::arg("l1_weight_pack5_asm"),
          pybind11::arg("l1_scale"),
          pybind11::arg("rank_idx"),
          pybind11::arg("num_ranks"),
          pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("hidden"),
          pybind11::arg("symm_base_addr"),
          pybind11::arg("symm_x_span"),
          pybind11::arg("alignment") = 256,
          pybind11::arg("code_object_path") = "",
          pybind11::arg("l1_out_workspace") = std::nullopt,
          pybind11::arg("cumulative_local_expert_recv_stats") = std::nullopt,
          pybind11::arg("runtime_num_tokens") = std::nullopt,
          pybind11::arg("force_compact_prebuild") = false,
          pybind11::arg("capacity_num_tokens") = -1);
    m.def("k1_symm_fused_l1_v3_pack5", &k1_symm_fused_l1_v3_pack5,
          pybind11::arg("sym_buffer"),
          pybind11::arg("route_scratch"),
          pybind11::arg("l1_weight_pack5"),
          pybind11::arg("l1_scale"),
          pybind11::arg("rank_idx"),
          pybind11::arg("num_ranks"),
          pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("hidden"),
          pybind11::arg("alignment") = 256,
          pybind11::arg("backend") = "normal",
          pybind11::arg("l1_out_workspace") = std::nullopt,
          pybind11::arg("cumulative_local_expert_recv_stats") = std::nullopt,
          pybind11::arg("runtime_num_tokens") = std::nullopt,
          pybind11::arg("ll_block_m") = 32,
          pybind11::arg("ll_cus") = 64,
          pybind11::arg("ll_asm_compatible_layout") = false,
          pybind11::arg("capacity_num_tokens") = -1,
          pybind11::arg("enable_start_rank_barrier") = false,
          pybind11::arg("tail_done_counter") = std::nullopt,
          pybind11::arg("graph_runtime_num_tokens_out") = std::nullopt,
          pybind11::arg("graph_tail_signal_generation_out") = std::nullopt);
    m.def("k1_graph_flag_reset_layout", &k1_graph_flag_reset_layout,
          pybind11::arg("num_ranks"),
          pybind11::arg("num_experts"),
          pybind11::arg("num_max_tokens_per_rank"),
          pybind11::arg("num_tokens"),
          pybind11::arg("num_topk"),
          pybind11::arg("hidden"),
          pybind11::arg("l1_rows"),
          pybind11::arg("alignment") = 256);
}
