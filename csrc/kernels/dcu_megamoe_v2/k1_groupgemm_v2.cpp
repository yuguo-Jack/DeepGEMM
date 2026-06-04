#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <deep_gemm/comm/mega_moe_dcu.cuh>
#include <deep_gemm/layout/mega_moe_dcu.cuh>

#ifndef DCU_MEGAMOE_V2_KERNEL_ONLY
#define HIP_CHECK(expr)                                                          \
    do {                                                                        \
        hipError_t _err = (expr);                                                \
        if (_err != hipSuccess) {                                                \
            std::cerr << "HIP error " << hipGetErrorString(_err)                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                        \
        }                                                                       \
    } while (0)

enum class Mode {
    kC,
    kLowLatencyC,
    kLowLatencySymmStageC,
    kCSymmStage,
};

struct Options {
    Mode mode = Mode::kC;
    int m = 8192;
    int n = 4096;
    int k = 4096;
    int experts = 32;
    int tokens = 0;
    int topk = 6;
    int symm_ranks = 8;
    int symm_devices = 1;
    int rank_idx = 0;
    int valid_rows_per_expert = 192;
    int warmup = 5;
    int repeat = 20;
    int check = 0;
    int touch_check = 0;
    int print_shape = 0;
    int measure_rounds = 1;
    int c_tile_n = 256;
    int c_lowlat_pack = 0;
    int c_stage_n_group = 4;
    int c_row_stage = 0;
    int k3_rowptr = 0;
    int k3_combine = 0;
    int k3_combine_linear = 0;
    int k3_copy_stage = 0;
    int k3_copy_workers = 16;
    int k3_tail_reduce = 0;
    int ll_cus = 64;
    int ll_block_m = 32;
    int ones = 0;
    int debug_values = 0;
    int n_pattern = 0;
    int realistic_values = 1;
    float input_value_scale = 0.02f;
    float weight_value_scale = 0.02f;
    double allowed_max_abs = 1.0e-3;
};

static int parse_int(const char* s) {
    return std::atoi(s);
}

static float parse_float(const char* s) {
    return std::strtof(s, nullptr);
}

static double parse_double(const char* s) {
    return std::strtod(s, nullptr);
}

static Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(2);
            }
            return argv[++i];
        };

        if (key == "--mode") {
            const std::string value(need_value("--mode"));
            if (value == "c-ll" || value == "ll-c") {
                opt.mode = Mode::kLowLatencyC;
                opt.c_tile_n = 64;
            } else if (value == "c-ll-symm-stage" || value == "ll-symm-stage") {
                opt.mode = Mode::kLowLatencySymmStageC;
                opt.c_tile_n = 64;
            } else if (value == "c-symm-stage") {
                opt.mode = Mode::kCSymmStage;
                opt.c_tile_n = 256;
                opt.c_lowlat_pack = 1;
            } else if (value == "c" || value == "c-mt256") {
                opt.mode = Mode::kC;
                opt.c_tile_n = 256;
            } else {
                std::cerr << "Unknown mode: " << value << std::endl;
                std::exit(2);
            }
        } else if (key == "--m") {
            opt.m = parse_int(need_value("--m"));
        } else if (key == "--n") {
            opt.n = parse_int(need_value("--n"));
        } else if (key == "--k") {
            opt.k = parse_int(need_value("--k"));
        } else if (key == "--experts") {
            opt.experts = parse_int(need_value("--experts"));
        } else if (key == "--rows-per-expert") {
            opt.valid_rows_per_expert = parse_int(need_value("--rows-per-expert"));
            opt.tokens = 0;
        } else if (key == "--tokens") {
            opt.tokens = parse_int(need_value("--tokens"));
        } else if (key == "--topk") {
            opt.topk = parse_int(need_value("--topk"));
        } else if (key == "--symm-ranks") {
            opt.symm_ranks = parse_int(need_value("--symm-ranks"));
        } else if (key == "--symm-devices") {
            opt.symm_devices = parse_int(need_value("--symm-devices"));
        } else if (key == "--rank-idx") {
            opt.rank_idx = parse_int(need_value("--rank-idx"));
        } else if (key == "--warmup") {
            opt.warmup = parse_int(need_value("--warmup"));
        } else if (key == "--repeat") {
            opt.repeat = parse_int(need_value("--repeat"));
        } else if (key == "--check") {
            opt.check = parse_int(need_value("--check"));
        } else if (key == "--touch-check") {
            opt.touch_check = parse_int(need_value("--touch-check"));
        } else if (key == "--print-shape") {
            opt.print_shape = parse_int(need_value("--print-shape"));
        } else if (key == "--measure-rounds") {
            opt.measure_rounds = parse_int(need_value("--measure-rounds"));
        } else if (key == "--c-tile-n") {
            opt.c_tile_n = parse_int(need_value("--c-tile-n"));
        } else if (key == "--c-lowlat-pack") {
            opt.c_lowlat_pack = parse_int(need_value("--c-lowlat-pack"));
        } else if (key == "--c-stage-n-group") {
            opt.c_stage_n_group = parse_int(need_value("--c-stage-n-group"));
        } else if (key == "--c-row-stage") {
            opt.c_row_stage = parse_int(need_value("--c-row-stage"));
        } else if (key == "--k3-rowptr") {
            opt.k3_rowptr = parse_int(need_value("--k3-rowptr"));
        } else if (key == "--k3-combine") {
            opt.k3_combine = parse_int(need_value("--k3-combine"));
        } else if (key == "--k3-combine-linear") {
            opt.k3_combine_linear = parse_int(need_value("--k3-combine-linear"));
        } else if (key == "--k3-copy-stage") {
            opt.k3_copy_stage = parse_int(need_value("--k3-copy-stage"));
        } else if (key == "--k3-copy-workers") {
            opt.k3_copy_workers = parse_int(need_value("--k3-copy-workers"));
        } else if (key == "--k3-tail-reduce") {
            opt.k3_tail_reduce = parse_int(need_value("--k3-tail-reduce"));
        } else if (key == "--ll-cus") {
            opt.ll_cus = parse_int(need_value("--ll-cus"));
        } else if (key == "--ll-block-m") {
            opt.ll_block_m = parse_int(need_value("--ll-block-m"));
        } else if (key == "--ones") {
            opt.ones = parse_int(need_value("--ones"));
        } else if (key == "--debug-values") {
            opt.debug_values = parse_int(need_value("--debug-values"));
        } else if (key == "--n-pattern") {
            opt.n_pattern = parse_int(need_value("--n-pattern"));
        } else if (key == "--realistic-values") {
            opt.realistic_values = parse_int(need_value("--realistic-values"));
        } else if (key == "--input-value-scale") {
            opt.input_value_scale = parse_float(need_value("--input-value-scale"));
        } else if (key == "--weight-value-scale") {
            opt.weight_value_scale = parse_float(need_value("--weight-value-scale"));
        } else if (key == "--allowed-max-abs") {
            opt.allowed_max_abs = parse_double(need_value("--allowed-max-abs"));
        } else {
            std::cerr << "Unknown option: " << key << std::endl;
            std::exit(2);
        }
    }
    if (opt.tokens > 0) {
        opt.valid_rows_per_expert =
            (opt.tokens * opt.topk + opt.experts - 1) / opt.experts;
    }
    if (opt.c_tile_n != 64 && opt.c_tile_n != 256) {
        std::cerr << "Expected --c-tile-n to be 64 or 256" << std::endl;
        std::exit(2);
    }
    return opt;
}

template <typename T>
static std::vector<T> marlin2_k64_n256_n16_transposed_weight(
    const std::vector<T>& w,
    int experts,
    int n,
    int k) {
    constexpr int k_outer_tile = 64;
    constexpr int n_outer_tile = 256;
    constexpr int k_segment = 16;
    constexpr int n_inner_tile = 16;
    const int k_outer = k / k_outer_tile;
    const int n_outer = n / n_outer_tile;
    std::vector<T> out(w.size());
    auto src_idx = [n, k](int e, int row, int col) {
        return static_cast<int64_t>(e) * n * k +
               static_cast<int64_t>(row) * k + col;
    };

    int64_t dst = 0;
    for (int e = 0; e < experts; ++e)
        for (int ko = 0; ko < k_outer; ++ko)
            for (int no = 0; no < n_outer; ++no)
                for (int ni16 = 0; ni16 < n_outer_tile / n_inner_tile; ++ni16)
                    for (int ks = 0; ks < k_outer_tile / k_segment; ++ks)
                        for (int ni = 0; ni < n_inner_tile; ++ni) {
                            const int src_ni = (ni & 3) * 4 + (ni >> 2);
                            for (int ki = 0; ki < k_segment; ++ki)
                                out[dst++] = w[src_idx(
                                    e,
                                    no * n_outer_tile + ni16 * n_inner_tile + src_ni,
                                    ko * k_outer_tile + ks * k_segment + ki)];
                        }
    return out;
}
#endif  // DCU_MEGAMOE_V2_KERNEL_ONLY

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using float32x2_t = float __attribute__((ext_vector_type(2)));
using float32x4_t = float __attribute__((ext_vector_type(4)));
using bf16x4_t = uint16_t __attribute__((ext_vector_type(4)));

__device__ int32x2_t llvm_amdgcn_raw_buffer_load_i32x2(
    int32x4_t resource,
    int voffset,
    int soffset,
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ int32_t llvm_amdgcn_raw_buffer_load_i32(
    int32x4_t resource,
    int voffset,
    int soffset,
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ int32x4_t llvm_amdgcn_raw_buffer_load_i32x4(
    int32x4_t resource,
    int voffset,
    int soffset,
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ void llvm_amdgcn_raw_buffer_store_ui16(
    uint16_t vdata,
    int32x4_t resource,
    int voffset,
    int soffset,
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16");

union Pack128 {
    int32x4_t v4;
    int32x2_t v2[2];
};

__device__ static inline uint16_t f32_to_bf16_bits_device(float value) {
#if defined(__gfx938__)
    return __builtin_hcu_cvt_bf16_f32(value, false, false);
#else
    uint32_t bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits >> 16) & 0xffffu);
#endif
}

__device__ static inline float bf16_bits_to_f32_device(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ static inline uint32_t pack2_bf16_bits_device(
    uint16_t lo,
    uint16_t hi) {
    return static_cast<uint32_t>(lo) |
           (static_cast<uint32_t>(hi) << 16);
}

__device__ static inline uint32_t pack2_bf16_f32_device(
    float lo,
    float hi) {
#if defined(__gfx938__)
    auto packed = __builtin_hcu_cvt_pk_bf16_f32(0, lo, 0, hi, 0);
    union {
        decltype(packed) value;
        uint32_t bits;
    } caster;
    caster.value = packed;
    return caster.bits;
#else
    return pack2_bf16_bits_device(
        f32_to_bf16_bits_device(lo), f32_to_bf16_bits_device(hi));
#endif
}

__device__ static inline void accumulate_bf16x8_device(
    const uint4 packed,
    float& sum0,
    float& sum1,
    float& sum2,
    float& sum3,
    float& sum4,
    float& sum5,
    float& sum6,
    float& sum7) {
    sum0 += bf16_bits_to_f32_device(static_cast<uint16_t>(packed.x));
    sum1 += bf16_bits_to_f32_device(static_cast<uint16_t>(packed.x >> 16));
    sum2 += bf16_bits_to_f32_device(static_cast<uint16_t>(packed.y));
    sum3 += bf16_bits_to_f32_device(static_cast<uint16_t>(packed.y >> 16));
    sum4 += bf16_bits_to_f32_device(static_cast<uint16_t>(packed.z));
    sum5 += bf16_bits_to_f32_device(static_cast<uint16_t>(packed.z >> 16));
    sum6 += bf16_bits_to_f32_device(static_cast<uint16_t>(packed.w));
    sum7 += bf16_bits_to_f32_device(static_cast<uint16_t>(packed.w >> 16));
}

__device__ static inline uint4 reduce_full_topk6_bf16x8_device(
    const uint4* combine_vecs,
    const int64_t token_vec_base,
    const int64_t slot_stride_vecs) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
#pragma unroll
    for (int topk_slot = 0; topk_slot < 6; ++topk_slot) {
        const uint4 packed =
            combine_vecs[token_vec_base +
                         static_cast<int64_t>(topk_slot) * slot_stride_vecs];
        accumulate_bf16x8_device(
            packed, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
    }
    uint4 reduced;
    reduced.x = pack2_bf16_f32_device(sum0, sum1);
    reduced.y = pack2_bf16_f32_device(sum2, sum3);
    reduced.z = pack2_bf16_f32_device(sum4, sum5);
    reduced.w = pack2_bf16_f32_device(sum6, sum7);
    return reduced;
}

__device__ static inline int64_t marlin_row_base_offset_device(
    int expert,
    int row,
    int rows,
    int k) {
    constexpr int n_tile = 16;
    constexpr int k_tile = 16;
    const int n_outer = row / n_tile;
    const int n_inner = row - n_outer * n_tile;
    return static_cast<int64_t>(expert) * rows * k +
           static_cast<int64_t>(n_outer) * (k / k_tile) * n_tile * k_tile +
           static_cast<int64_t>(n_inner) * k_tile;
}

__device__ static inline int marlin_k_offset_device(int k_idx) {
    constexpr int n_tile = 16;
    constexpr int k_tile = 16;
    const int k_outer = k_idx / k_tile;
    const int k_inner = k_idx - k_outer * k_tile;
    return k_outer * n_tile * k_tile + k_inner;
}

__device__ static inline int deepgemm_stage_order_device(
    int stage_iter,
    int wg_n_tile,
    int num_k_stages) {
    if (wg_n_tile >= 8)
        return stage_iter;
    const int half_stages = num_k_stages >> 1;
    return half_stages > 0 ? (stage_iter ^ half_stages) : stage_iter;
}

__device__ static inline float32x4_t mmac_fp8_device(
    int32x2_t a,
    int32x2_t b,
    float32x4_t c) {
#if defined(__gfx938__)
    return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a, b, c, false, false);
#else
    return c;
#endif
}

__device__ static inline float vec4_get_device(float32x4_t value, int slot) {
    return slot == 0 ? value.x : (slot == 1 ? value.y : (slot == 2 ? value.z : value.w));
}

__device__ static inline float32x4_t shuffle_acc_lane_device(
    float32x4_t value,
    int source_lane) {
    return float32x4_t{
        __shfl(vec4_get_device(value, 0), source_lane),
        __shfl(vec4_get_device(value, 1), source_lane),
        __shfl(vec4_get_device(value, 2), source_lane),
        __shfl(vec4_get_device(value, 3), source_lane)};
}

__device__ static inline float i32_bits_to_f32_device(int32_t bits) {
    return *reinterpret_cast<float*>(&bits);
}

__device__ static inline float32x2_t pk_mul_f32_device(float32x2_t lhs, float32x2_t rhs) {
    float32x2_t result;
    asm volatile("v_pk_mul_f32 %0, %1, %2\n"
                 : "=v"(result)
                 : "v"(lhs), "v"(rhs));
    return result;
}

__device__ static inline int32x4_t make_buffer_resource_device(const uint8_t* ptr) {
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    int32x4_t resource{0, 0, static_cast<int32_t>(0x80000000u), 0x00020000};
    resource[0] = static_cast<int32_t>(addr);
    resource[1] = static_cast<int32_t>(addr >> 32);
    return resource;
}

__device__ static inline void buffer_load_lds_16b_device(
    const int32x4_t resource,
    uint8_t* lds_base,
    int lds_byte_offset,
    int global_byte_offset) {
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + lds_byte_offset);
    auto* lds_ptr = (__attribute__((address_space(3))) int*)lds_addr;
    __builtin_amdgcn_raw_buffer_load_lds(
        resource, lds_ptr, 16, global_byte_offset, 0, 0, 0);
    return;
}

__device__ static inline void buffer_load_lds_4b_device(
    const int32x4_t resource,
    uint8_t* lds_base,
    int lds_byte_offset,
    int global_byte_offset) {
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + lds_byte_offset);
    auto* lds_ptr = (__attribute__((address_space(3))) int*)lds_addr;
    __builtin_amdgcn_raw_buffer_load_lds(
        resource, lds_ptr, 4, global_byte_offset, 0, 0, 0);
    return;
}

__device__ static inline void buffer_load_lds_8x16b_m0_device(
    const int32x4_t resource,
    uint8_t* lds_base,
    int lds_first_byte_offset,
    uint32_t global_offset0,
    uint32_t global_stride = 0x20000u) {
    using LdsIntPtr = __attribute__((address_space(3))) int*;
    const auto lds_ptr = (LdsIntPtr)(lds_base + lds_first_byte_offset);
    const uint32_t lds_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds_ptr));
    const uint32_t global_offset1 = global_offset0 + 1u * global_stride;
    const uint32_t global_offset2 = global_offset0 + 2u * global_stride;
    const uint32_t global_offset3 = global_offset0 + 3u * global_stride;
    const uint32_t global_offset4 = global_offset0 + 4u * global_stride;
    const uint32_t global_offset5 = global_offset0 + 5u * global_stride;
    const uint32_t global_offset6 = global_offset0 + 6u * global_stride;
    const uint32_t global_offset7 = global_offset0 + 7u * global_stride;
    uint32_t lds_addr_scalar;
    asm volatile("v_readfirstlane_b32 %0, %1\n\t"
                 : "=s"(lds_addr_scalar)
                 : "v"(lds_addr));
    asm volatile(
        "s_mov_b32 m0, %0\n\t"
        "s_nop 0\n\t"
        "buffer_load_dwordx4 %1, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %2, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %3, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %4, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %5, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %6, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %7, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %8, %9, 0 offen lds\n\t"
        :
        : "s"(lds_addr_scalar),
          "v"(global_offset0),
          "v"(global_offset1),
          "v"(global_offset2),
          "v"(global_offset3),
          "v"(global_offset4),
          "v"(global_offset5),
          "v"(global_offset6),
          "v"(global_offset7),
          "s"(resource)
        : "memory");
    return;
}

__device__ static inline void buffer_load_lds_8x16b_m0_stride800_device(
    const int32x4_t resource,
    uint8_t* lds_base,
    int lds_first_byte_offset,
    uint32_t global_offset0) {
    using LdsIntPtr = __attribute__((address_space(3))) int*;
    const auto lds_ptr = (LdsIntPtr)(lds_base + lds_first_byte_offset);
    const uint32_t lds_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds_ptr));
    const uint32_t global_offset1 = global_offset0 + 1u * 0x800u;
    const uint32_t global_offset2 = global_offset0 + 2u * 0x800u;
    const uint32_t global_offset3 = global_offset0 + 3u * 0x800u;
    const uint32_t global_offset4 = global_offset0 + 4u * 0x800u;
    const uint32_t global_offset5 = global_offset0 + 5u * 0x800u;
    const uint32_t global_offset6 = global_offset0 + 6u * 0x800u;
    const uint32_t global_offset7 = global_offset0 + 7u * 0x800u;
    uint32_t lds_addr_scalar;
    asm volatile("v_readfirstlane_b32 %0, %1\n\t"
                 : "=s"(lds_addr_scalar)
                 : "v"(lds_addr));
    asm volatile(
        "s_mov_b32 m0, %0\n\t"
        "s_nop 0\n\t"
        "buffer_load_dwordx4 %1, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %2, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %3, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %4, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %5, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %6, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %7, %9, 0 offen lds\n\t"
        "s_add_u32 m0, m0, 0x1000\n\t"
        "buffer_load_dwordx4 %8, %9, 0 offen lds\n\t"
        :
        : "s"(lds_addr_scalar),
          "v"(global_offset0),
          "v"(global_offset1),
          "v"(global_offset2),
          "v"(global_offset3),
          "v"(global_offset4),
          "v"(global_offset5),
          "v"(global_offset6),
          "v"(global_offset7),
          "s"(resource)
        : "memory");
    return;
}

__device__ static inline void buffer_load_lds_2x16b_m0_device(
    const int32x4_t resource,
    uint8_t* lds_base,
    int lds_first_byte_offset,
    uint32_t global_offset0,
    uint32_t global_stride = 0x20000u) {
    const uint32_t global_offset1 = global_offset0 + global_stride;
    buffer_load_lds_16b_device(
        resource, lds_base, lds_first_byte_offset,
        static_cast<int>(global_offset0));
    buffer_load_lds_16b_device(
        resource, lds_base, lds_first_byte_offset + 4096,
        static_cast<int>(global_offset1));
    return;
}

__device__ static inline int32x2_t buffer_load_fp8_b64_device(
    const int32x4_t resource,
    int global_byte_offset) {
    return llvm_amdgcn_raw_buffer_load_i32x2(resource, global_byte_offset, 0, 0);
}

__device__ static inline float buffer_load_f32_device(
    const int32x4_t resource,
    int global_byte_offset) {
    return i32_bits_to_f32_device(
        llvm_amdgcn_raw_buffer_load_i32(resource, global_byte_offset, 0, 0));
}

__device__ static inline Pack128 buffer_load_fp8_b128_pack_device(
    const int32x4_t resource,
    int global_byte_offset) {
    Pack128 value;
    value.v4 = llvm_amdgcn_raw_buffer_load_i32x4(resource, global_byte_offset, 0, 0);
    return value;
}

__device__ static inline void buffer_store_bf16_device(
    const int32x4_t resource,
    uint16_t value,
    int global_byte_offset) {
    llvm_amdgcn_raw_buffer_store_ui16(value, resource, global_byte_offset, 0, 0);
    return;
}

__device__ static inline float ds_read_f32_device(
    const uint8_t* lds_base,
    int byte_offset) {
    uint32_t bits;
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + byte_offset);
    asm volatile("ds_read_b32 %0, %1\n"
                 : "=v"(bits)
                 : "v"(static_cast<uint32_t>(lds_addr)));
    return __uint_as_float(bits);
}

template <int kOffset>
__device__ static inline int32x2_t ds_read_b64_offset_device(
    const uint8_t* lds_base,
    int byte_offset) {
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + (byte_offset & ~7));
    int32x2_t value;
    asm volatile(
        "ds_read_b64 %0, %1 offset:%2\n"
        : "=v"(value)
        : "v"(static_cast<uint32_t>(lds_addr)), "n"(kOffset));
    return value;
}

template <int kOffset>
__device__ static inline Pack128 ds_read_b128_offset_device(
    const uint8_t* lds_base,
    int byte_offset) {
    Pack128 value;
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + byte_offset);
    asm volatile(
        "ds_read_b128 %0, %1 offset:%2\n"
        : "=v"(value.v4)
        : "v"(static_cast<uint32_t>(lds_addr)), "n"(kOffset));
    return value;
}

__device__ static inline void ds_read16_b128_wait6_device(
    const uint8_t* lds_base,
    int byte_offset,
    Pack128& a0,
    Pack128& a1,
    Pack128& a2,
    Pack128& a3,
    Pack128& a4,
    Pack128& a5,
    Pack128& a6,
    Pack128& a7,
    Pack128& a8,
    Pack128& a9,
    Pack128& a10,
    Pack128& a11,
    Pack128& a12,
    Pack128& a13,
    Pack128& a14,
    Pack128& a15) {
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + byte_offset);
    asm volatile(
        "ds_read_b128 %[a0], %[addr] offset:0\n\t"
        "ds_read_b128 %[a1], %[addr] offset:2048\n\t"
        "ds_read_b128 %[a2], %[addr] offset:4096\n\t"
        "ds_read_b128 %[a3], %[addr] offset:6144\n\t"
        "ds_read_b128 %[a4], %[addr] offset:8192\n\t"
        "ds_read_b128 %[a5], %[addr] offset:10240\n\t"
        "ds_read_b128 %[a6], %[addr] offset:12288\n\t"
        "ds_read_b128 %[a7], %[addr] offset:14336\n\t"
        "ds_read_b128 %[a8], %[addr] offset:16384\n\t"
        "ds_read_b128 %[a9], %[addr] offset:18432\n\t"
        "ds_read_b128 %[a10], %[addr] offset:20480\n\t"
        "ds_read_b128 %[a11], %[addr] offset:22528\n\t"
        "ds_read_b128 %[a12], %[addr] offset:24576\n\t"
        "ds_read_b128 %[a13], %[addr] offset:26624\n\t"
        "ds_read_b128 %[a14], %[addr] offset:28672\n\t"
        "ds_read_b128 %[a15], %[addr] offset:30720\n\t"
        "s_waitcnt lgkmcnt(6)\n\t"
        : [a0] "=&v"(a0.v4), [a1] "=&v"(a1.v4),
          [a2] "=&v"(a2.v4), [a3] "=&v"(a3.v4),
          [a4] "=&v"(a4.v4), [a5] "=&v"(a5.v4),
          [a6] "=&v"(a6.v4), [a7] "=&v"(a7.v4),
          [a8] "=&v"(a8.v4), [a9] "=&v"(a9.v4),
          [a10] "=&v"(a10.v4), [a11] "=&v"(a11.v4),
          [a12] "=&v"(a12.v4), [a13] "=&v"(a13.v4),
          [a14] "=&v"(a14.v4), [a15] "=&v"(a15.v4)
        : [addr] "v"(static_cast<uint32_t>(lds_addr))
        : "memory");
    return;
}

__device__ static inline void ds_read4_b128_wait0_device(
    const uint8_t* lds_base,
    int byte_offset,
    Pack128& a0,
    Pack128& a1,
    Pack128& a2,
    Pack128& a3) {
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + byte_offset);
    asm volatile(
        "ds_read_b128 %[a0], %[addr] offset:0\n\t"
        "ds_read_b128 %[a1], %[addr] offset:2048\n\t"
        "ds_read_b128 %[a2], %[addr] offset:4096\n\t"
        "ds_read_b128 %[a3], %[addr] offset:6144\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : [a0] "=&v"(a0.v4), [a1] "=&v"(a1.v4),
          [a2] "=&v"(a2.v4), [a3] "=&v"(a3.v4)
        : [addr] "v"(static_cast<uint32_t>(lds_addr))
        : "memory");
    return;
}

__device__ static inline void wait_vmem_lds_store_device() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    return;
}

__device__ static inline void wait_lds_read_device() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    return;
}

__device__ static inline void wait_lds_read_overlap_device() {
    asm volatile("s_waitcnt lgkmcnt(6)");
    __builtin_amdgcn_sched_barrier(0);
    return;
}

__device__ static inline void set_mmac_priority_high_device() {
    asm volatile("s_setprio 1\n\t" ::: "memory");
    return;
}

__device__ static inline void set_mmac_priority_normal_device() {
    asm volatile("s_setprio 0\n\t" ::: "memory");
    return;
}

__device__ static inline void block_barrier_device() {
    asm volatile("s_barrier\n" ::: "memory");
    return;
}

__device__ static inline void store_acc_fragment_scaled_unmasked_device(
    const int32x4_t out_resource,
    float32x4_t acc,
    int hidden,
    int n,
    int token_base,
    float weight_scale,
    float32x2_t x_pair0,
    float32x2_t x_pair1) {
    const float32x2_t w_pair{weight_scale, weight_scale};

    float32x2_t c_pair0{vec4_get_device(acc, 0), vec4_get_device(acc, 1)};
    float32x2_t scale_pair0 = pk_mul_f32_device(w_pair, x_pair0);
    float32x2_t out_pair0 = pk_mul_f32_device(scale_pair0, c_pair0);
    buffer_store_bf16_device(
        out_resource,
        f32_to_bf16_bits_device(out_pair0.x),
        static_cast<int>((static_cast<int64_t>(token_base + 0) * n + hidden) * 2));
    buffer_store_bf16_device(
        out_resource,
        f32_to_bf16_bits_device(out_pair0.y),
        static_cast<int>((static_cast<int64_t>(token_base + 4) * n + hidden) * 2));

    float32x2_t c_pair1{vec4_get_device(acc, 2), vec4_get_device(acc, 3)};
    float32x2_t scale_pair1 = pk_mul_f32_device(w_pair, x_pair1);
    float32x2_t out_pair1 = pk_mul_f32_device(scale_pair1, c_pair1);
    buffer_store_bf16_device(
        out_resource,
        f32_to_bf16_bits_device(out_pair1.x),
        static_cast<int>((static_cast<int64_t>(token_base + 8) * n + hidden) * 2));
    buffer_store_bf16_device(
        out_resource,
        f32_to_bf16_bits_device(out_pair1.y),
        static_cast<int>((static_cast<int64_t>(token_base + 12) * n + hidden) * 2));
    return;
}

__device__ static inline void store_acc_fragment_scaled_masked_device(
    const int32x4_t out_resource,
    float32x4_t acc,
    int hidden,
    int n,
    int token_base,
    float weight_scale,
    float32x2_t x_pair0,
    float32x2_t x_pair1,
    int valid_mask) {
    const float32x2_t w_pair{weight_scale, weight_scale};

    float32x2_t c_pair0{vec4_get_device(acc, 0), vec4_get_device(acc, 1)};
    float32x2_t scale_pair0 = pk_mul_f32_device(w_pair, x_pair0);
    float32x2_t out_pair0 = pk_mul_f32_device(scale_pair0, c_pair0);
    if (valid_mask & 0x1) {
        buffer_store_bf16_device(
            out_resource,
            f32_to_bf16_bits_device(out_pair0.x),
            static_cast<int>((static_cast<int64_t>(token_base + 0) * n + hidden) * 2));
    }
    if (valid_mask & 0x2) {
        buffer_store_bf16_device(
            out_resource,
            f32_to_bf16_bits_device(out_pair0.y),
            static_cast<int>((static_cast<int64_t>(token_base + 4) * n + hidden) * 2));
    }

    float32x2_t c_pair1{vec4_get_device(acc, 2), vec4_get_device(acc, 3)};
    float32x2_t scale_pair1 = pk_mul_f32_device(w_pair, x_pair1);
    float32x2_t out_pair1 = pk_mul_f32_device(scale_pair1, c_pair1);
    if (valid_mask & 0x4) {
        buffer_store_bf16_device(
            out_resource,
            f32_to_bf16_bits_device(out_pair1.x),
            static_cast<int>((static_cast<int64_t>(token_base + 8) * n + hidden) * 2));
    }
    if (valid_mask & 0x8) {
        buffer_store_bf16_device(
            out_resource,
            f32_to_bf16_bits_device(out_pair1.y),
            static_cast<int>((static_cast<int64_t>(token_base + 12) * n + hidden) * 2));
    }
    return;
}

__device__ static inline void rowptr_store_bf16_device(
    const int64_t* row_output_ptrs,
    int row,
    int hidden,
    uint16_t value) {
    const int64_t row_addr = row_output_ptrs[row];
    if (row_addr == 0)
        return;
    reinterpret_cast<uint16_t*>(row_addr)[hidden] = value;
}

__device__ static inline void rowaddr_store_bf16_device(
    int64_t row_addr,
    int hidden,
    uint16_t value) {
    if (row_addr == 0)
        return;
    reinterpret_cast<uint16_t*>(row_addr)[hidden] = value;
}

__device__ static inline void store_acc_fragment_scaled_unmasked_rowptr_device(
    const int64_t* row_output_ptrs,
    float32x4_t acc,
    int hidden,
    int token_base,
    float weight_scale,
    float32x2_t x_pair0,
    float32x2_t x_pair1) {
    const float32x2_t w_pair{weight_scale, weight_scale};

    float32x2_t c_pair0{vec4_get_device(acc, 0), vec4_get_device(acc, 1)};
    float32x2_t scale_pair0 = pk_mul_f32_device(w_pair, x_pair0);
    float32x2_t out_pair0 = pk_mul_f32_device(scale_pair0, c_pair0);
    rowptr_store_bf16_device(
        row_output_ptrs, token_base + 0, hidden,
        f32_to_bf16_bits_device(out_pair0.x));
    rowptr_store_bf16_device(
        row_output_ptrs, token_base + 4, hidden,
        f32_to_bf16_bits_device(out_pair0.y));

    float32x2_t c_pair1{vec4_get_device(acc, 2), vec4_get_device(acc, 3)};
    float32x2_t scale_pair1 = pk_mul_f32_device(w_pair, x_pair1);
    float32x2_t out_pair1 = pk_mul_f32_device(scale_pair1, c_pair1);
    rowptr_store_bf16_device(
        row_output_ptrs, token_base + 8, hidden,
        f32_to_bf16_bits_device(out_pair1.x));
    rowptr_store_bf16_device(
        row_output_ptrs, token_base + 12, hidden,
        f32_to_bf16_bits_device(out_pair1.y));
    return;
}

__device__ static inline void store_acc_fragment_scaled_unmasked_rowaddr_device(
    int64_t row_addr0,
    int64_t row_addr4,
    int64_t row_addr8,
    int64_t row_addr12,
    float32x4_t acc,
    int hidden,
    float weight_scale,
    float32x2_t x_pair0,
    float32x2_t x_pair1) {
    const float32x2_t w_pair{weight_scale, weight_scale};

    float32x2_t c_pair0{vec4_get_device(acc, 0), vec4_get_device(acc, 1)};
    float32x2_t scale_pair0 = pk_mul_f32_device(w_pair, x_pair0);
    float32x2_t out_pair0 = pk_mul_f32_device(scale_pair0, c_pair0);
    rowaddr_store_bf16_device(
        row_addr0, hidden, f32_to_bf16_bits_device(out_pair0.x));
    rowaddr_store_bf16_device(
        row_addr4, hidden, f32_to_bf16_bits_device(out_pair0.y));

    float32x2_t c_pair1{vec4_get_device(acc, 2), vec4_get_device(acc, 3)};
    float32x2_t scale_pair1 = pk_mul_f32_device(w_pair, x_pair1);
    float32x2_t out_pair1 = pk_mul_f32_device(scale_pair1, c_pair1);
    rowaddr_store_bf16_device(
        row_addr8, hidden, f32_to_bf16_bits_device(out_pair1.x));
    rowaddr_store_bf16_device(
        row_addr12, hidden, f32_to_bf16_bits_device(out_pair1.y));
    return;
}

__device__ static inline void store_acc_fragment_scaled_masked_rowaddr_device(
    int64_t row_addr0,
    int64_t row_addr4,
    int64_t row_addr8,
    int64_t row_addr12,
    float32x4_t acc,
    int hidden,
    float weight_scale,
    float32x2_t x_pair0,
    float32x2_t x_pair1,
    int valid_mask) {
    const float32x2_t w_pair{weight_scale, weight_scale};

    float32x2_t c_pair0{vec4_get_device(acc, 0), vec4_get_device(acc, 1)};
    float32x2_t scale_pair0 = pk_mul_f32_device(w_pair, x_pair0);
    float32x2_t out_pair0 = pk_mul_f32_device(scale_pair0, c_pair0);
    if (valid_mask & 0x1) {
        rowaddr_store_bf16_device(
            row_addr0, hidden, f32_to_bf16_bits_device(out_pair0.x));
    }
    if (valid_mask & 0x2) {
        rowaddr_store_bf16_device(
            row_addr4, hidden, f32_to_bf16_bits_device(out_pair0.y));
    }

    float32x2_t c_pair1{vec4_get_device(acc, 2), vec4_get_device(acc, 3)};
    float32x2_t scale_pair1 = pk_mul_f32_device(w_pair, x_pair1);
    float32x2_t out_pair1 = pk_mul_f32_device(scale_pair1, c_pair1);
    if (valid_mask & 0x4) {
        rowaddr_store_bf16_device(
            row_addr8, hidden, f32_to_bf16_bits_device(out_pair1.x));
    }
    if (valid_mask & 0x8) {
        rowaddr_store_bf16_device(
            row_addr12, hidden, f32_to_bf16_bits_device(out_pair1.y));
    }
    return;
}

__device__ static inline void store_acc_fragment_scaled_masked_rowptr_device(
    const int64_t* row_output_ptrs,
    float32x4_t acc,
    int hidden,
    int token_base,
    float weight_scale,
    float32x2_t x_pair0,
    float32x2_t x_pair1,
    int valid_mask) {
    const float32x2_t w_pair{weight_scale, weight_scale};

    float32x2_t c_pair0{vec4_get_device(acc, 0), vec4_get_device(acc, 1)};
    float32x2_t scale_pair0 = pk_mul_f32_device(w_pair, x_pair0);
    float32x2_t out_pair0 = pk_mul_f32_device(scale_pair0, c_pair0);
    if (valid_mask & 0x1) {
        rowptr_store_bf16_device(
            row_output_ptrs, token_base + 0, hidden,
            f32_to_bf16_bits_device(out_pair0.x));
    }
    if (valid_mask & 0x2) {
        rowptr_store_bf16_device(
            row_output_ptrs, token_base + 4, hidden,
            f32_to_bf16_bits_device(out_pair0.y));
    }

    float32x2_t c_pair1{vec4_get_device(acc, 2), vec4_get_device(acc, 3)};
    float32x2_t scale_pair1 = pk_mul_f32_device(w_pair, x_pair1);
    float32x2_t out_pair1 = pk_mul_f32_device(scale_pair1, c_pair1);
    if (valid_mask & 0x4) {
        rowptr_store_bf16_device(
            row_output_ptrs, token_base + 8, hidden,
            f32_to_bf16_bits_device(out_pair1.x));
    }
    if (valid_mask & 0x8) {
        rowptr_store_bf16_device(
            row_output_ptrs, token_base + 12, hidden,
            f32_to_bf16_bits_device(out_pair1.y));
    }
    return;
}

__device__ static inline int row_in_expert_valid_mask_device(
    int valid_rows_per_expert,
    int row_in_expert) {
    int mask = 0;
    if (row_in_expert + 0 < valid_rows_per_expert)
        mask |= 0x1;
    if (row_in_expert + 4 < valid_rows_per_expert)
        mask |= 0x2;
    if (row_in_expert + 8 < valid_rows_per_expert)
        mask |= 0x4;
    if (row_in_expert + 12 < valid_rows_per_expert)
        mask |= 0x8;
    return mask;
}

__device__ static inline bool v2_symm_source_for_grouped_row(
    int grouped_row,
    int rows_aligned_per_expert,
    int valid_rows_per_expert,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int& source_rank,
    int& source_token) {
    const int local_experts = num_global_experts / num_ranks;
    const int local_expert = grouped_row / rows_aligned_per_expert;
    const int row_in_expert =
        grouped_row - local_expert * rows_aligned_per_expert;
    if (row_in_expert < 0 || row_in_expert >= valid_rows_per_expert)
        return false;
    const int global_expert = rank_idx * local_experts + local_expert;
    const int routes_per_rank = num_max_tokens_per_rank * num_topk;
    const int64_t route_linear =
        static_cast<int64_t>(row_in_expert) * num_global_experts +
        global_expert;
    if (route_linear < 0 ||
        route_linear >= static_cast<int64_t>(num_ranks) * routes_per_rank)
        return false;
    source_rank = static_cast<int>(route_linear / routes_per_rank);
    const int local_route =
        static_cast<int>(route_linear - static_cast<int64_t>(source_rank) *
                                         routes_per_rank);
    source_token = local_route / num_topk;
    return true;
}

__device__ static inline void v2_device_grid_barrier(
    int32_t* barrier,
    int expected_blocks);

__device__ static inline int v2_effective_num_tokens(
    const int32_t* runtime_num_tokens_ptr,
    int runtime_num_tokens,
    int num_max_tokens_per_rank) {
    int value = runtime_num_tokens >= 0
                    ? runtime_num_tokens
                    : runtime_num_tokens_ptr[0];
    if (value < 0)
        value = 0;
    if (value > num_max_tokens_per_rank)
        value = num_max_tokens_per_rank;
    return value;
}

template <int kTileRows, int kTileN, bool kLowlatWeightLayout = false,
          bool kUseSymmRowStage = false,
          int kNGroup = 1, int kProblemN = 4096, int kProblemK = 4096,
          bool kUseRowPtrs = false, bool kUseK3CopyStage = false,
          bool kUseDirectSymmLoad = false>
__global__ __launch_bounds__(768, 1) void
V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16(
    hip_bfloat16* out,
    const uint8_t* x,
    const uint8_t* weight_marlin,
    const float* x_scale,
    const float* w_scale,
    const int32_t* row_expert,
    int m,
    int n,
    int k,
    int rows_aligned_per_expert,
    int valid_rows_per_expert,
    uint8_t* local_sym_buffer,
    int32_t* grid_barrier,
    int launch_epoch,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    const int64_t* row_output_ptrs,
    int k3_copy_workers = 0,
    int k3_tail_reduce = 0,
    const uint8_t* k3_local_topk_mask = nullptr,
    hip_bfloat16* k3_tail_out = nullptr,
    const int32_t* k3_tail_tokens = nullptr,
    int k3_tail_token_count = 0,
    int32_t* route_scratch_i32 = nullptr,
    float* route_weights_out = nullptr,
    int32_t* row_expert_out = nullptr,
    int32_t* output_index = nullptr,
    int64_t* row_combine_ptrs_out = nullptr,
    uint8_t* local_topk_mask = nullptr,
    int32_t* tail_tokens = nullptr,
    int32_t* cumulative_local_expert_recv_stats = nullptr) {
    static_assert(kTileRows == 64 || kTileRows == 256, "unsupported row tile");
    static_assert(kTileN == 64 || kTileN == 256, "unsupported tile N");
    static_assert(kNGroup == 1 || kNGroup == 2 || kNGroup == 4 ||
                      kNGroup == 8,
                  "unsupported large C N group");
    static_assert(kProblemN == 4096, "V2 large C currently supports N=4096");
    static_assert(kProblemK == 4096 || kProblemK == 2048,
                  "V2 large C currently supports K=4096 or K=2048");
    constexpr int kTileM = kTileRows;
    constexpr int kStageK = 128;
    constexpr int kLdsStages = 2;
    constexpr int kWaveNGroups = kTileM / 32;
    constexpr int kComputeWaves = kWaveNGroups;
    constexpr int kVecBytes = 16;
    constexpr int kSingleStageBytes = kTileN * kStageK;
    constexpr int kSingleStageVecs = kSingleStageBytes / kVecBytes;
    constexpr int kTotalStageVecs = kLdsStages * kSingleStageVecs;
    constexpr uint32_t kWeightExpertStride =
        static_cast<uint32_t>(kProblemN * kProblemK);
    constexpr uint32_t kKpackN0Stride =
        static_cast<uint32_t>(kProblemK * 16);
    constexpr uint32_t kKpackM0Stride =
        static_cast<uint32_t>(kKpackN0Stride * 2);
    constexpr uint32_t kPack5K64Stride =
        static_cast<uint32_t>((kProblemN / 256) * 0x4000u);
    if (n != kProblemN || k != kProblemK)
        return;

    __shared__ uint4 lds_stage[kTotalStageVecs];
    auto* lds_bytes = reinterpret_cast<uint8_t*>(lds_stage);
#define K1_BLOCK_BARRIER()                                                      \
    do {                                                                        \
        block_barrier_device();                                                 \
    } while (0)

    const int wave_id = static_cast<int>(threadIdx.x >> 6);
    const int lane = static_cast<int>(threadIdx.x & 63);
    const int lane_m = lane & 15;
    const int lane_n = lane & 15;
    const int lane_k = (lane >> 4) * 8;
    const int lane_group = lane >> 4;
    const int c_n_base = lane_group;

    int k3_copy_rows = 0;
    if constexpr (kUseK3CopyStage) {
        const int requested_workers =
            k3_copy_workers < 1 ? static_cast<int>(gridDim.x)
                                : k3_copy_workers;
        k3_copy_rows =
            (requested_workers + static_cast<int>(gridDim.x) - 1) /
            static_cast<int>(gridDim.x);
    }
    const int compute_grid_y =
        static_cast<int>(gridDim.y) - k3_copy_rows;
    if constexpr (kUseK3CopyStage) {
        if (static_cast<int>(blockIdx.y) < k3_copy_rows) {
            constexpr int kCopyBf16PerVec = 8;
            const int worker_id =
                static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) +
                static_cast<int>(blockIdx.x);
            const int requested_workers =
                k3_copy_workers < 1 ? static_cast<int>(gridDim.x)
                                    : k3_copy_workers;
            const int worker_count =
                requested_workers >
                        k3_copy_rows * static_cast<int>(gridDim.x)
                    ? k3_copy_rows * static_cast<int>(gridDim.x)
                    : requested_workers;
            if (worker_id >= worker_count)
                return;
            const int total_tiles = compute_grid_y * static_cast<int>(gridDim.x);
            volatile int32_t* copy_done_count =
                reinterpret_cast<volatile int32_t*>(grid_barrier + total_tiles);
            volatile int32_t* copy_done_epoch =
                reinterpret_cast<volatile int32_t*>(
                    grid_barrier + total_tiles + 1);
            if (k3_tail_reduce != 0 && worker_id == 0 && threadIdx.x == 0) {
                *copy_done_count = 0;
                __threadfence_system();
                *copy_done_epoch = launch_epoch;
            }
            if (k3_tail_reduce != 0) {
                while (*copy_done_epoch != launch_epoch) {
                }
            }
            uintptr_t local_combine_begin = 0;
            uintptr_t local_combine_end = 0;
            if (k3_tail_reduce != 0) {
                auto local_sections = deep_gemm::mega::get_sections(
                    local_sym_buffer, num_ranks, num_global_experts,
                    num_max_tokens_per_rank, num_topk, kProblemN);
                local_combine_begin =
                    reinterpret_cast<uintptr_t>(local_sections.combine);
                local_combine_end =
                    local_combine_begin +
                    static_cast<uintptr_t>(num_topk) *
                        static_cast<uintptr_t>(num_max_tokens_per_rank) *
                        static_cast<uintptr_t>(kProblemN) *
                        static_cast<uintptr_t>(sizeof(uint16_t));
            }
            for (int tile = worker_id; tile < total_tiles;
                 tile += worker_count) {
                volatile int32_t* flag =
                    reinterpret_cast<volatile int32_t*>(
                        grid_barrier + tile);
                while (*flag != launch_epoch) {
                }
                const int tile_y = tile / static_cast<int>(gridDim.x);
                const int tile_x =
                    tile - tile_y * static_cast<int>(gridDim.x);
                const int row_base = tile_y * kTileM;
                const int col_base = tile_x * kTileN;
                const int copy_tile_expert =
                    row_base / rows_aligned_per_expert;
                const int copy_row_in_expert_base =
                    row_base - copy_tile_expert * rows_aligned_per_expert;
                int copy_active_rows = kTileM;
                if (copy_row_in_expert_base >= valid_rows_per_expert) {
                    copy_active_rows = 0;
                } else if (copy_row_in_expert_base + kTileM >
                           valid_rows_per_expert) {
                    copy_active_rows =
                        valid_rows_per_expert - copy_row_in_expert_base;
                }
                const int vecs_per_row = kTileN / kCopyBf16PerVec;
                const int total_vecs = copy_active_rows * vecs_per_row;
                for (int linear = static_cast<int>(threadIdx.x);
                     linear < total_vecs;
                     linear += static_cast<int>(blockDim.x)) {
                    const int row_in_tile = linear / vecs_per_row;
                    const int vec_col = linear - row_in_tile * vecs_per_row;
                    const int row = row_base + row_in_tile;
                    if (row >= m)
                        continue;
                    const int64_t row_addr = row_output_ptrs[row];
                    if (row_addr == 0)
                        continue;
                    if (k3_tail_reduce != 0) {
                        const uintptr_t dst_addr =
                            static_cast<uintptr_t>(row_addr);
                        if (dst_addr < local_combine_begin ||
                            dst_addr >= local_combine_end)
                            continue;
                    }
                    const auto* src = reinterpret_cast<const uint4*>(
                        out + static_cast<int64_t>(row) * kProblemN +
                        col_base);
                    auto* dst = reinterpret_cast<uint4*>(
                        reinterpret_cast<uint16_t*>(row_addr) + col_base);
                    dst[vec_col] = src[vec_col];
                }
            }
            if (k3_tail_reduce != 0) {
                wait_vmem_lds_store_device();
                block_barrier_device();
                if (threadIdx.x == 0) {
                    __threadfence_system();
                    atomicAdd(grid_barrier + total_tiles, 1);
                }
                while (*copy_done_count < worker_count) {
                }

                constexpr int kReduceBf16PerVec = 8;
                auto local_sections = deep_gemm::mega::get_sections(
                    local_sym_buffer, num_ranks, num_global_experts,
                    num_max_tokens_per_rank, num_topk, kProblemN);
                const int vecs_per_token = kProblemN / kReduceBf16PerVec;
                int tail_reduce_tokens = k3_tail_token_count;
                if (tail_reduce_tokens < 0) {
                    tail_reduce_tokens = v2_effective_num_tokens(
                        local_sections.num_tokens, runtime_num_tokens,
                        num_max_tokens_per_rank);
                }
                const int64_t total_reduce_vecs =
                    static_cast<int64_t>(
                        k3_tail_tokens != nullptr ? tail_reduce_tokens
                                                  : num_max_tokens_per_rank) *
                    vecs_per_token;
                auto* out_vec = reinterpret_cast<uint4*>(
                    k3_tail_out != nullptr ? k3_tail_out : out);
                const auto* combine_vecs =
                    reinterpret_cast<const uint4*>(local_sections.combine);
                const int64_t slot_stride_vecs =
                    static_cast<int64_t>(num_max_tokens_per_rank) *
                    vecs_per_token;
                const bool dense_identity_tail_tokens =
                    (k3_tail_tokens == nullptr) ||
                    (k3_tail_token_count >= 0 &&
                     runtime_num_tokens == k3_tail_token_count);
                const bool fixed_full_topk6_tail_reduce =
                    (k3_tail_tokens != nullptr) && (num_topk == 6) &&
                    (k3_tail_token_count >= 0) &&
                    (runtime_num_tokens == k3_tail_token_count);
                for (int64_t task =
                         (static_cast<int64_t>(worker_id) * blockDim.x) +
                         threadIdx.x;
                     task < total_reduce_vecs;
                     task += static_cast<int64_t>(worker_count) *
                             blockDim.x) {
                    const int reduce_token_idx =
                        static_cast<int>(task / vecs_per_token);
                    const int vec_idx =
                        static_cast<int>(
                            task -
                            static_cast<int64_t>(reduce_token_idx) *
                                vecs_per_token);
                    const int token_idx =
                        dense_identity_tail_tokens
                            ? reduce_token_idx
                            : (k3_tail_tokens != nullptr
                                   ? k3_tail_tokens[reduce_token_idx]
                                   : reduce_token_idx);
                    const int64_t token_vec_base =
                        dense_identity_tail_tokens
                            ? task
                            : static_cast<int64_t>(token_idx) *
                                      vecs_per_token +
                                  vec_idx;
                    const int64_t out_task = token_vec_base;
                    if (fixed_full_topk6_tail_reduce) {
                        out_vec[out_task] = reduce_full_topk6_bf16x8_device(
                            combine_vecs, token_vec_base, slot_stride_vecs);
                        continue;
                    }
                    uint32_t slot_mask = 0;
                    if (k3_local_topk_mask != nullptr) {
                        slot_mask = static_cast<uint32_t>(
                            k3_local_topk_mask[token_idx]);
                    } else {
                        const int local_experts =
                            num_global_experts / num_ranks;
                        const int first_expert = rank_idx * local_experts;
                        const int last_expert = first_expert + local_experts;
                        for (int topk_slot = 0; topk_slot < num_topk;
                             ++topk_slot) {
                            const int64_t global_expert =
                                local_sections.topk_idx[
                                    static_cast<int64_t>(token_idx) *
                                        num_topk +
                                    topk_slot];
                            if (global_expert >= first_expert &&
                                global_expert < last_expert) {
                                slot_mask |= (1u << topk_slot);
                            }
                        }
                    }
                    if (slot_mask == 0) {
                        if (k3_tail_out != nullptr)
                            continue;
                        uint4 zero;
                        zero.x = 0;
                        zero.y = 0;
                        zero.z = 0;
                        zero.w = 0;
                        out_vec[out_task] = zero;
                        continue;
                    }
                    if ((slot_mask & (slot_mask - 1u)) == 0) {
                        int topk_slot = 0;
                        while ((slot_mask & (1u << topk_slot)) == 0)
                            ++topk_slot;
                        out_vec[out_task] =
                            combine_vecs[token_vec_base +
                                         static_cast<int64_t>(topk_slot) *
                                             slot_stride_vecs];
                        continue;
                    }
                    if (num_topk == 6 && slot_mask == 0x3fu) {
                        out_vec[out_task] = reduce_full_topk6_bf16x8_device(
                            combine_vecs, token_vec_base, slot_stride_vecs);
                        continue;
                    }
                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;
                    float sum3 = 0.0f;
                    float sum4 = 0.0f;
                    float sum5 = 0.0f;
                    float sum6 = 0.0f;
                    float sum7 = 0.0f;
                    for (int topk_slot = 0; topk_slot < num_topk;
                         ++topk_slot) {
                        if ((slot_mask & (1u << topk_slot)) == 0)
                            continue;
                        const uint4 packed =
                            combine_vecs
                                [token_vec_base +
                                 static_cast<int64_t>(topk_slot) *
                                     slot_stride_vecs];
                        accumulate_bf16x8_device(
                            packed, sum0, sum1, sum2, sum3, sum4, sum5, sum6,
                            sum7);
                    }
                    uint4 reduced;
                    reduced.x = pack2_bf16_f32_device(sum0, sum1);
                    reduced.y = pack2_bf16_f32_device(sum2, sum3);
                    reduced.z = pack2_bf16_f32_device(sum4, sum5);
                    reduced.w = pack2_bf16_f32_device(sum6, sum7);
                    out_vec[out_task] = reduced;
                }
            }
            return;
        }
    }

    const int wave_n_group = wave_id;
    const int compute_block_y =
        static_cast<int>(blockIdx.y) - k3_copy_rows;
    const int tile_token = compute_block_y * kTileM;
    const int token_base = tile_token + wave_n_group * 32;
    const int tile_expert = tile_token / rows_aligned_per_expert;
    const int tile_row_in_expert =
        tile_token - tile_expert * rows_aligned_per_expert;
    const int wave_row_in_expert =
        tile_row_in_expert + wave_n_group * 32;
    const bool wave_has_valid_rows =
        wave_row_in_expert < valid_rows_per_expert;
    (void)m;
    (void)row_expert;

    int32_t* row_stage_barrier = grid_barrier;
    int32_t* metadata_stage_count = nullptr;
    int32_t* metadata_stage_epoch = nullptr;
    if constexpr (kUseSymmRowStage) {
        row_stage_barrier = grid_barrier + 2;
        metadata_stage_count =
            row_stage_barrier + 2 * static_cast<int>(gridDim.y);
        metadata_stage_epoch =
            row_stage_barrier + 3 * static_cast<int>(gridDim.y);
    }

    int32_t* symm_src_ranks = nullptr;
    int32_t* symm_src_tokens = nullptr;
    if constexpr (kUseSymmRowStage) {
        if (route_scratch_i32 != nullptr) {
            constexpr int kMetadataExperts = 32;
            int32_t* symm_counts = route_scratch_i32;
            symm_src_ranks = route_scratch_i32 + kMetadataExperts;
            symm_src_tokens =
                symm_src_ranks +
                static_cast<int64_t>(kMetadataExperts) *
                    rows_aligned_per_expert;

            uint8_t** peer_sym_buffers =
                deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);
            const int local_experts = num_global_experts / num_ranks;
            const int first_expert = rank_idx * local_experts;
            const int last_expert = first_expert + local_experts;
            const int routes_per_rank = num_max_tokens_per_rank * num_topk;
            const bool metadata_tile_block = tile_row_in_expert == 0;
            const bool metadata_owner_block =
                metadata_tile_block && static_cast<int>(blockIdx.x) == 0;
            if (metadata_owner_block && tile_expert < kMetadataExperts &&
                local_sym_buffer != nullptr && grid_barrier != nullptr &&
                num_ranks > 1) {
                if (tile_expert == 0) {
                    int** peer_signal_buffers =
                        deep_gemm::mega::dcu_peer_signal_ptrs(
                            local_sym_buffer, num_ranks);
                    deep_gemm::mega::mega_moe_rank_barrier(
                        peer_signal_buffers, rank_idx, num_ranks);
                    if (threadIdx.x == 0) {
                        __threadfence();
                        grid_barrier[0] = launch_epoch;
                    }
                } else {
                    volatile int32_t* sync_flag =
                        reinterpret_cast<volatile int32_t*>(grid_barrier);
                    while (*sync_flag != launch_epoch) {
                    }
                }
                block_barrier_device();
            }
            if (metadata_tile_block) {
                const int metadata_expert = tile_expert;
                if (metadata_owner_block &&
                    metadata_expert < kMetadataExperts) {
                    if (threadIdx.x == 0) {
                        symm_counts[metadata_expert] = 0;
                    }
                    for (int row_in_expert = static_cast<int>(threadIdx.x);
                         row_in_expert < rows_aligned_per_expert;
                         row_in_expert += static_cast<int>(blockDim.x)) {
                        const int row =
                            metadata_expert * rows_aligned_per_expert +
                            row_in_expert;
                        symm_src_ranks[row] = -1;
                        symm_src_tokens[row] = 0;
                        if (route_weights_out != nullptr)
                            route_weights_out[row] = 0.0f;
                        if (row_expert_out != nullptr)
                            row_expert_out[row] = metadata_expert;
                        if (row_combine_ptrs_out != nullptr)
                            row_combine_ptrs_out[row] = 0;
                    }
                }
                if (metadata_owner_block && metadata_expert == 0 &&
                    (local_topk_mask != nullptr || tail_tokens != nullptr)) {
                    auto local_sections = deep_gemm::mega::get_sections(
                        peer_sym_buffers[rank_idx], num_ranks,
                        num_global_experts, num_max_tokens_per_rank, num_topk,
                        kProblemK);
                    int local_effective_tokens = v2_effective_num_tokens(
                        local_sections.num_tokens, runtime_num_tokens,
                        num_max_tokens_per_rank);
                    for (int token_idx = static_cast<int>(threadIdx.x);
                         token_idx < num_max_tokens_per_rank;
                         token_idx += static_cast<int>(blockDim.x)) {
                        uint8_t mask = 0;
                        if (token_idx < local_effective_tokens) {
                            for (int topk_slot = 0; topk_slot < num_topk;
                                 ++topk_slot) {
                                const int64_t route_offset =
                                    static_cast<int64_t>(token_idx) * num_topk +
                                    topk_slot;
                                const int64_t expert =
                                    local_sections.topk_idx[route_offset];
                                const float route_weight =
                                    local_sections.topk_weights[route_offset];
                                if (expert >= 0 &&
                                    expert < num_global_experts &&
                                    route_weight != 0.0f) {
                                    mask |=
                                        static_cast<uint8_t>(1u << topk_slot);
                                }
                            }
                        }
                        if (local_topk_mask != nullptr)
                            local_topk_mask[token_idx] = mask;
                        if (tail_tokens != nullptr)
                            tail_tokens[token_idx] = token_idx;
                    }
                }
                if (metadata_owner_block &&
                    metadata_expert < kMetadataExperts &&
                    threadIdx.x == 0) {
                    metadata_stage_count[static_cast<int>(blockIdx.y)] = 0;
                    __threadfence();
                    metadata_stage_epoch[static_cast<int>(blockIdx.y)] =
                        launch_epoch;
                }
                block_barrier_device();
                if (!metadata_owner_block &&
                    metadata_expert < kMetadataExperts) {
                    volatile int32_t* metadata_ready =
                        reinterpret_cast<volatile int32_t*>(
                            metadata_stage_epoch +
                            static_cast<int>(blockIdx.y));
                    while (*metadata_ready != launch_epoch) {
                    }
                }
                block_barrier_device();

                if (metadata_expert < kMetadataExperts) {
                    for (int source_rank = 0; source_rank < num_ranks;
                         ++source_rank) {
                        auto sections = deep_gemm::mega::get_sections(
                            peer_sym_buffers[source_rank], num_ranks,
                            num_global_experts, num_max_tokens_per_rank,
                            num_topk, kProblemK);
                        const int effective_tokens = v2_effective_num_tokens(
                            sections.num_tokens, runtime_num_tokens,
                            num_max_tokens_per_rank);
                        const int valid_routes = effective_tokens * num_topk;
                        for (int local_route =
                                 static_cast<int>(blockIdx.x) *
                                     static_cast<int>(blockDim.x) +
                                 static_cast<int>(threadIdx.x);
                             local_route < valid_routes;
                             local_route += static_cast<int>(gridDim.x) *
                                            static_cast<int>(blockDim.x)) {
                            const int token_idx = local_route / num_topk;
                            const int topk_slot =
                                local_route - token_idx * num_topk;
                            const int64_t expert =
                                sections.topk_idx
                                    [static_cast<int64_t>(token_idx) *
                                         num_topk +
                                     topk_slot];
                            const float route_weight =
                                sections.topk_weights
                                    [static_cast<int64_t>(token_idx) *
                                         num_topk +
                                     topk_slot];
                            const bool accepted =
                                expert >= first_expert &&
                                expert < last_expert &&
                                route_weight != 0.0f;
                            if (!accepted) {
                                if (metadata_expert == 0 &&
                                    output_index != nullptr) {
                                    output_index
                                        [static_cast<int64_t>(source_rank) *
                                             routes_per_rank +
                                         local_route] = -1;
                                }
                                continue;
                            }
                            const int route_local_expert =
                                static_cast<int>(expert) - first_expert;
                            if (route_local_expert != metadata_expert)
                                continue;
                            const int row_in_expert =
                                atomicAdd(symm_counts + metadata_expert, 1);
                            const int64_t route_linear =
                                static_cast<int64_t>(source_rank) *
                                    routes_per_rank +
                                local_route;
                            if (row_in_expert >= rows_aligned_per_expert) {
                                if (output_index != nullptr)
                                    output_index[route_linear] = -1;
                                continue;
                            }
                            const int64_t row =
                                static_cast<int64_t>(metadata_expert) *
                                    rows_aligned_per_expert +
                                row_in_expert;
                            symm_src_ranks[row] = source_rank;
                            symm_src_tokens[row] = token_idx;
                            if (route_weights_out != nullptr)
                                route_weights_out[row] = route_weight;
                            if (row_expert_out != nullptr)
                                row_expert_out[row] = metadata_expert;
                            if (output_index != nullptr)
                                output_index[route_linear] =
                                    static_cast<int32_t>(row);
                            if (row_combine_ptrs_out != nullptr) {
                                const int64_t partial_row =
                                    static_cast<int64_t>(topk_slot) *
                                        num_max_tokens_per_rank +
                                    token_idx;
                                row_combine_ptrs_out[row] =
                                    static_cast<int64_t>(
                                        reinterpret_cast<uintptr_t>(
                                            sections.combine +
                                            partial_row * kProblemN));
                            }
                        }
                        if (metadata_expert == 0 && output_index != nullptr) {
                            for (int local_route =
                                     valid_routes +
                                     static_cast<int>(threadIdx.x);
                                 local_route < routes_per_rank;
                                 local_route += static_cast<int>(blockDim.x)) {
                                output_index
                                    [static_cast<int64_t>(source_rank) *
                                         routes_per_rank +
                                     local_route] = -1;
                            }
                        }
                    }
                    block_barrier_device();
                    __threadfence();
                    if (threadIdx.x == 0)
                        atomicAdd(metadata_stage_count +
                                      static_cast<int>(blockIdx.y),
                                  1);
                    volatile int32_t* metadata_count =
                        reinterpret_cast<volatile int32_t*>(
                            metadata_stage_count +
                            static_cast<int>(blockIdx.y));
                    while (*metadata_count < static_cast<int>(gridDim.x)) {
                    }
                    block_barrier_device();
                    if (metadata_owner_block && threadIdx.x == 0 &&
                        symm_counts[metadata_expert] >
                            rows_aligned_per_expert) {
                        symm_counts[metadata_expert] =
                            rows_aligned_per_expert;
                    }
                    block_barrier_device();
                    if (metadata_owner_block && threadIdx.x == 0 &&
                        cumulative_local_expert_recv_stats != nullptr) {
                        atomicAdd(cumulative_local_expert_recv_stats +
                                      metadata_expert,
                                  symm_counts[metadata_expert]);
                    }
                }
                if (metadata_owner_block &&
                    metadata_expert < kMetadataExperts && threadIdx.x == 0) {
                    __threadfence();
                    const int first_tile_y =
                        (metadata_expert * rows_aligned_per_expert) / kTileM;
                    const int tile_count =
                        (rows_aligned_per_expert + kTileM - 1) / kTileM;
                    for (int tile_iter = 0; tile_iter < tile_count;
                         ++tile_iter) {
                        const int target_y = first_tile_y + tile_iter;
                        if (target_y < static_cast<int>(gridDim.y)) {
                            row_stage_barrier[target_y] = 0;
                        }
                    }
                    __threadfence();
                    for (int tile_iter = 0; tile_iter < tile_count;
                         ++tile_iter) {
                        const int target_y = first_tile_y + tile_iter;
                        if (target_y < static_cast<int>(gridDim.y)) {
                            row_stage_barrier[static_cast<int>(gridDim.y) +
                                              target_y] = launch_epoch;
                        }
                    }
                }
            }
        }
    }

    const int token_col0 = token_base + lane_n;
    const int token_col1 = token_col0 + 16;
    if constexpr (kUseSymmRowStage) {
        constexpr int kStageVecBytes = 16;
        constexpr int kStageVecsPerRow = kProblemK / kStageVecBytes;
        constexpr int kStageRows = kTileM;
        const int32x4_t zero_vec{0, 0, 0, 0};
        uint8_t** peer_sym_buffers =
            deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);

        if constexpr (kUseSymmRowStage) {
            if (route_scratch_i32 == nullptr &&
                static_cast<int>(blockIdx.x) == 0 && threadIdx.x == 0) {
                row_stage_barrier[static_cast<int>(blockIdx.y)] = 0;
                __threadfence();
                row_stage_barrier[static_cast<int>(gridDim.y) +
                                  static_cast<int>(blockIdx.y)] =
                    launch_epoch;
            }
            volatile int32_t* row_epoch =
                reinterpret_cast<volatile int32_t*>(
                row_stage_barrier + static_cast<int>(gridDim.y) +
                    static_cast<int>(blockIdx.y));
            while (*row_epoch != launch_epoch) {
            }
        }

        if constexpr (!kUseDirectSymmLoad) {
            auto* staged_vecs =
                reinterpret_cast<int32x4_t*>(const_cast<uint8_t*>(x));
            const int stage_linear_begin =
                static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x);
            const int stage_linear_stride =
                static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
            for (int linear = stage_linear_begin;
                 linear < kStageRows * kStageVecsPerRow;
                 linear += stage_linear_stride) {
                const int local_row = linear / kStageVecsPerRow;
                const int vec_col = linear - local_row * kStageVecsPerRow;
                const int grouped_row = tile_token + local_row;
                const int row_in_expert =
                    grouped_row - tile_expert * rows_aligned_per_expert;
                if (row_in_expert < 0 ||
                    row_in_expert >= rows_aligned_per_expert)
                    continue;
                int32x4_t value = zero_vec;
                if (row_in_expert < valid_rows_per_expert) {
                    int source_rank = 0;
                    int source_token = 0;
                    bool has_source = false;
                    if (route_scratch_i32 != nullptr) {
                        source_rank = symm_src_ranks[grouped_row];
                        source_token = symm_src_tokens[grouped_row];
                        has_source = source_rank >= 0;
                    } else {
                        has_source = v2_symm_source_for_grouped_row(
                            grouped_row, rows_aligned_per_expert,
                            valid_rows_per_expert, rank_idx, num_ranks,
                            num_global_experts, num_max_tokens_per_rank,
                            num_topk, source_rank, source_token);
                    }
                    if (has_source) {
                        auto sections = deep_gemm::mega::get_sections(
                            peer_sym_buffers[source_rank], num_ranks,
                            num_global_experts, num_max_tokens_per_rank,
                            num_topk, kProblemK);
                        value = reinterpret_cast<const int32x4_t*>(
                            sections.x +
                            static_cast<int64_t>(source_token) *
                                kProblemK)[vec_col];
                    }
                }
                staged_vecs[static_cast<int64_t>(grouped_row) *
                                kStageVecsPerRow +
                            vec_col] = value;
            }
        }
        if (!kUseSymmRowStage || static_cast<int>(blockIdx.x) == 0) {
            for (int local_row = static_cast<int>(threadIdx.x);
                 local_row < kStageRows;
                 local_row += static_cast<int>(blockDim.x)) {
                const int grouped_row = tile_token + local_row;
                const int row_in_expert =
                    grouped_row - tile_expert * rows_aligned_per_expert;
                if (row_in_expert < 0 ||
                    row_in_expert >= rows_aligned_per_expert)
                    continue;
                float scale = 1.0e-4f / 448.0f;
                if (row_in_expert < valid_rows_per_expert) {
                    int source_rank = 0;
                    int source_token = 0;
                    bool has_source = false;
                    if (route_scratch_i32 != nullptr) {
                        source_rank = symm_src_ranks[grouped_row];
                        source_token = symm_src_tokens[grouped_row];
                        has_source = source_rank >= 0;
                    } else {
                        has_source = v2_symm_source_for_grouped_row(
                            grouped_row, rows_aligned_per_expert,
                            valid_rows_per_expert, rank_idx, num_ranks,
                            num_global_experts, num_max_tokens_per_rank,
                            num_topk, source_rank, source_token);
                    }
                    if (has_source) {
                        auto sections = deep_gemm::mega::get_sections(
                            peer_sym_buffers[source_rank], num_ranks,
                            num_global_experts, num_max_tokens_per_rank,
                            num_topk, kProblemK);
                        scale = sections.x_sf[source_token];
                    }
                }
                const_cast<float*>(x_scale)[grouped_row] = scale;
            }
        }
        block_barrier_device();
        wait_vmem_lds_store_device();
        __threadfence_system();
        block_barrier_device();
        if constexpr (kUseSymmRowStage) {
            if (threadIdx.x == 0)
                atomicAdd(row_stage_barrier + static_cast<int>(blockIdx.y), 1);
            volatile int32_t* row_count =
                reinterpret_cast<volatile int32_t*>(
                    row_stage_barrier + static_cast<int>(blockIdx.y));
            while (*row_count < static_cast<int>(gridDim.x)) {
            }
        }
        block_barrier_device();
    }
    for (int n_group_iter = 0; n_group_iter < kNGroup; ++n_group_iter) {
    const int tile_hidden =
        (static_cast<int>(blockIdx.x) * kNGroup + n_group_iter) * kTileN;
    if (tile_hidden >= kProblemN)
        continue;
    const int hidden_order_tile = tile_hidden >> 8;
    const int hidden_base = tile_hidden;
    auto lds_weight_write_offset = [](int loader_linear, int segment) {
        return loader_linear * kVecBytes + segment * 4096;
    };
#define K1_ACC_ZERO {0.0f, 0.0f, 0.0f, 0.0f}
    float32x4_t acc00 K1_ACC_ZERO;
    float32x4_t acc10 K1_ACC_ZERO;
    float32x4_t acc20 K1_ACC_ZERO;
    float32x4_t acc30 K1_ACC_ZERO;
    float32x4_t acc40 K1_ACC_ZERO;
    float32x4_t acc50 K1_ACC_ZERO;
    float32x4_t acc60 K1_ACC_ZERO;
    float32x4_t acc70 K1_ACC_ZERO;
    float32x4_t acc02 K1_ACC_ZERO;
    float32x4_t acc12 K1_ACC_ZERO;
    float32x4_t acc22 K1_ACC_ZERO;
    float32x4_t acc32 K1_ACC_ZERO;
    float32x4_t acc42 K1_ACC_ZERO;
    float32x4_t acc52 K1_ACC_ZERO;
    float32x4_t acc62 K1_ACC_ZERO;
    float32x4_t acc72 K1_ACC_ZERO;
    float32x4_t acc01 K1_ACC_ZERO;
    float32x4_t acc11 K1_ACC_ZERO;
    float32x4_t acc21 K1_ACC_ZERO;
    float32x4_t acc31 K1_ACC_ZERO;
    float32x4_t acc41 K1_ACC_ZERO;
    float32x4_t acc51 K1_ACC_ZERO;
    float32x4_t acc61 K1_ACC_ZERO;
    float32x4_t acc71 K1_ACC_ZERO;
    float32x4_t acc03 K1_ACC_ZERO;
    float32x4_t acc13 K1_ACC_ZERO;
    float32x4_t acc23 K1_ACC_ZERO;
    float32x4_t acc33 K1_ACC_ZERO;
    float32x4_t acc43 K1_ACC_ZERO;
    float32x4_t acc53 K1_ACC_ZERO;
    float32x4_t acc63 K1_ACC_ZERO;
    float32x4_t acc73 K1_ACC_ZERO;
#undef K1_ACC_ZERO
    const int32x4_t x_resource = make_buffer_resource_device(x);
    const int32x4_t weight_resource = make_buffer_resource_device(weight_marlin);
    const uint8_t* direct_x0 = nullptr;
    const uint8_t* direct_x1 = nullptr;
    if constexpr (kUseDirectSymmLoad) {
        uint8_t** peer_sym_buffers =
            deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);
        auto resolve_direct_x = [&](int grouped_row) -> const uint8_t* {
            const int row_in_expert =
                grouped_row - tile_expert * rows_aligned_per_expert;
            if (row_in_expert < 0 ||
                row_in_expert >= valid_rows_per_expert)
                return nullptr;
            int source_rank = 0;
            int source_token = 0;
            bool has_source = false;
            if (route_scratch_i32 != nullptr) {
                source_rank = symm_src_ranks[grouped_row];
                source_token = symm_src_tokens[grouped_row];
                has_source = source_rank >= 0;
            } else {
                has_source = v2_symm_source_for_grouped_row(
                    grouped_row, rows_aligned_per_expert,
                    valid_rows_per_expert, rank_idx, num_ranks,
                    num_global_experts, num_max_tokens_per_rank, num_topk,
                    source_rank, source_token);
            }
            if (!has_source)
                return nullptr;
            auto sections = deep_gemm::mega::get_sections(
                peer_sym_buffers[source_rank], num_ranks, num_global_experts,
                num_max_tokens_per_rank, num_topk, kProblemK);
            return sections.x + static_cast<int64_t>(source_token) * kProblemK;
        };
        direct_x0 = resolve_direct_x(token_col0);
        direct_x1 = resolve_direct_x(token_col1);
    }
    auto load_x_pack = [&](const uint8_t* direct_ptr,
                           int grouped_row,
                           int phase_k) -> Pack128 {
        Pack128 value;
        if constexpr (kUseDirectSymmLoad) {
            if (direct_ptr == nullptr) {
                value.v4 = int32x4_t{0, 0, 0, 0};
            } else {
                value.v4 = reinterpret_cast<const int32x4_t*>(
                    direct_ptr + phase_k)[0];
            }
        } else {
            value = buffer_load_fp8_b128_pack_device(
                x_resource,
                static_cast<int>(static_cast<int64_t>(grouped_row) *
                                     kProblemK +
                                 phase_k));
        }
        return value;
    };
#define K1_PREFETCH_A_STAGE(STAGE_BASE, K_STAGE)                                \
        do {                                                                    \
            const int loader_linear = (wave_id - kComputeWaves) * 64 + lane;    \
            if constexpr (kLowlatWeightLayout) {                                \
                const uint32_t physical_ni = static_cast<uint32_t>(loader_linear & 15); \
                const uint32_t k16_group =                                      \
                    static_cast<uint32_t>((loader_linear & 127) >> 4);          \
                const uint32_t global_offset0 =                                 \
                    static_cast<uint32_t>(tile_expert) * kWeightExpertStride +  \
                    (static_cast<uint32_t>(K_STAGE) >> 6) * kPack5K64Stride +   \
                    static_cast<uint32_t>(k16_group >> 2) * kPack5K64Stride +   \
                    static_cast<uint32_t>(tile_hidden >> 8) * 0x4000u +         \
                    static_cast<uint32_t>(loader_linear >> 7) * 0x400u +        \
                    static_cast<uint32_t>(k16_group & 3u) * 0x100u +            \
                    physical_ni * 16u;                                          \
                buffer_load_lds_8x16b_m0_stride800_device(                      \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + lds_weight_write_offset(loader_linear, 0),   \
                    global_offset0);                                            \
            } else {                                                            \
                const uint32_t global_offset0 =                                 \
                    static_cast<uint32_t>(tile_expert) * kWeightExpertStride +  \
                    static_cast<uint32_t>(tile_hidden >> 4) * kKpackN0Stride +  \
                    static_cast<uint32_t>(K_STAGE) * 16u +                      \
                    static_cast<uint32_t>(loader_linear >> 7) * kKpackN0Stride +\
                    static_cast<uint32_t>(loader_linear & 15) * 16u +           \
                    static_cast<uint32_t>((loader_linear & 127) >> 4) * 256u;   \
                buffer_load_lds_8x16b_m0_device(                                \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + lds_weight_write_offset(loader_linear, 0),   \
                    global_offset0, kKpackM0Stride);                            \
            }                                                                   \
        } while (0)
#define K1_PREFETCH_A_STAGE64(STAGE_BASE, K_STAGE)                              \
        do {                                                                    \
            const int loader_linear = (wave_id - kComputeWaves) * 64 + lane;    \
            if constexpr (kLowlatWeightLayout) {                                \
                const uint32_t physical_ni = static_cast<uint32_t>(loader_linear & 15); \
                const uint32_t k16_group =                                      \
                    static_cast<uint32_t>((loader_linear & 127) >> 4);          \
                const uint32_t global_offset0 =                                 \
                    static_cast<uint32_t>(tile_expert) * kWeightExpertStride +  \
                    (static_cast<uint32_t>(K_STAGE) >> 6) * kPack5K64Stride +   \
                    static_cast<uint32_t>(k16_group >> 2) * kPack5K64Stride +   \
                    static_cast<uint32_t>(tile_hidden >> 8) * 0x4000u +         \
                    static_cast<uint32_t>(loader_linear >> 7) * 0x400u +        \
                    static_cast<uint32_t>(k16_group & 3u) * 0x100u +            \
                    physical_ni * 16u;                                          \
                buffer_load_lds_16b_device(                                     \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + lds_weight_write_offset(loader_linear, 0),   \
                    static_cast<int>(global_offset0));                          \
                buffer_load_lds_16b_device(                                     \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + lds_weight_write_offset(loader_linear, 1),   \
                    static_cast<int>(global_offset0 + 0x800u));                 \
            } else {                                                            \
                const uint32_t global_offset0 =                                 \
                    static_cast<uint32_t>(tile_expert) * kWeightExpertStride +  \
                    static_cast<uint32_t>(tile_hidden >> 4) * kKpackN0Stride +  \
                    static_cast<uint32_t>(K_STAGE) * 16u +                      \
                    static_cast<uint32_t>(loader_linear >> 7) * kKpackN0Stride +\
                    static_cast<uint32_t>(loader_linear & 15) * 16u +           \
                    static_cast<uint32_t>((loader_linear & 127) >> 4) * 256u;   \
                buffer_load_lds_2x16b_m0_device(                                \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + lds_weight_write_offset(loader_linear, 0),   \
                    global_offset0, kKpackM0Stride);                            \
            }                                                                   \
        } while (0)
#define K1_PREFETCH_A_STAGE_ROW64(STAGE_BASE, K_STAGE)                          \
        do {                                                                    \
            const int loader_linear = (wave_id - kComputeWaves) * 64 + lane;    \
            if constexpr (kLowlatWeightLayout) {                                \
                const uint32_t physical_ni = static_cast<uint32_t>(loader_linear & 15); \
                const uint32_t k16_group =                                      \
                    static_cast<uint32_t>((loader_linear & 127) >> 4);          \
                const uint32_t global_offset0 =                                 \
                    static_cast<uint32_t>(tile_expert) * kWeightExpertStride +  \
                    (static_cast<uint32_t>(K_STAGE) >> 6) * kPack5K64Stride +   \
                    static_cast<uint32_t>(k16_group >> 2) * kPack5K64Stride +   \
                    static_cast<uint32_t>(tile_hidden >> 8) * 0x4000u +         \
                    static_cast<uint32_t>(k16_group & 3u) * 0x100u +            \
                    physical_ni * 16u;                                          \
                buffer_load_lds_8x16b_m0_stride800_device(                      \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + loader_linear * kVecBytes,                   \
                    global_offset0);                                            \
                buffer_load_lds_8x16b_m0_stride800_device(                      \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + loader_linear * kVecBytes + 2048,            \
                    global_offset0 + 0x400u);                                   \
            } else {                                                            \
                const uint32_t global_offset0 =                                 \
                    static_cast<uint32_t>(tile_expert) * kWeightExpertStride +  \
                    static_cast<uint32_t>(tile_hidden >> 4) * kKpackN0Stride +  \
                    static_cast<uint32_t>(K_STAGE) * 16u +                      \
                    static_cast<uint32_t>(loader_linear & 15) * 16u +           \
                    static_cast<uint32_t>((loader_linear & 127) >> 4) * 256u;   \
                buffer_load_lds_8x16b_m0_device(                                \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + loader_linear * kVecBytes,                   \
                    global_offset0, kKpackM0Stride);                            \
                buffer_load_lds_8x16b_m0_device(                                \
                    weight_resource, lds_bytes,                                 \
                    (STAGE_BASE) + loader_linear * kVecBytes + 2048,            \
                    global_offset0 + kKpackN0Stride, kKpackM0Stride);           \
            }                                                                   \
        } while (0)
    const int num_k_stages = kProblemK / kStageK;
    if (wave_id >= kComputeWaves) {
        for (int stage_iter = 0; stage_iter < num_k_stages; ++stage_iter) {
            const int stage_base = (stage_iter & 1) * kSingleStageBytes;
            const int k_stage =
                deepgemm_stage_order_device(
                    stage_iter, hidden_order_tile, num_k_stages) *
                kStageK;
            if constexpr (kTileRows == 64 && kTileN == 256) {
                K1_PREFETCH_A_STAGE_ROW64(stage_base, k_stage);
            } else if constexpr (kTileN == 64) {
                K1_PREFETCH_A_STAGE64(stage_base, k_stage);
            } else {
                K1_PREFETCH_A_STAGE(stage_base, k_stage);
            }
            wait_vmem_lds_store_device();
            if constexpr (kProblemK == 2048)
                __builtin_amdgcn_sched_barrier(0);
            K1_BLOCK_BARRIER();
        }
        const int scale_stage_base =
            ((((num_k_stages - 1) & 1) ^ 1) * kSingleStageBytes);
        const int scale_linear = (wave_id - kComputeWaves) * 64 + lane;
        const int32x4_t x_scale_loader_resource =
            make_buffer_resource_device(reinterpret_cast<const uint8_t*>(x_scale));
        const int32x4_t w_scale_loader_resource =
            make_buffer_resource_device(reinterpret_cast<const uint8_t*>(w_scale));
        const int64_t loader_weight_scale_base =
            static_cast<int64_t>(tile_expert) * kProblemN;
        if constexpr (kTileM == 256) {
            buffer_load_lds_4b_device(
                x_scale_loader_resource, lds_bytes,
                scale_stage_base + scale_linear * 4,
                static_cast<int>((tile_token + scale_linear) * 4));
        } else {
            if (scale_linear < kTileM) {
                buffer_load_lds_4b_device(
                    x_scale_loader_resource, lds_bytes,
                    scale_stage_base + scale_linear * 4,
                    static_cast<int>((tile_token + scale_linear) * 4));
            }
        }
        if constexpr (kTileN == 256) {
            buffer_load_lds_4b_device(
                w_scale_loader_resource, lds_bytes,
                scale_stage_base + 1024 + scale_linear * 4,
                static_cast<int>((loader_weight_scale_base + hidden_base +
                                  scale_linear) * 4));
            if constexpr (kTileM == 64) {
                buffer_load_lds_4b_device(
                    w_scale_loader_resource, lds_bytes,
                    scale_stage_base + 1024 + (scale_linear + 128) * 4,
                    static_cast<int>((loader_weight_scale_base + hidden_base +
                                      scale_linear + 128) * 4));
            }
        } else {
            if (scale_linear < kTileN) {
                buffer_load_lds_4b_device(
                    w_scale_loader_resource, lds_bytes,
                    scale_stage_base + 1024 + scale_linear * 4,
                    static_cast<int>((loader_weight_scale_base + hidden_base +
                                      scale_linear) * 4));
            }
        }
        wait_vmem_lds_store_device();
        if constexpr (kProblemK == 2048)
            __builtin_amdgcn_sched_barrier(0);
        K1_BLOCK_BARRIER();
    } else {
        K1_BLOCK_BARRIER();

#define K1_MMAC_ROWS_0_7_B0(A0, A1, A2, A3, A4, A5, A6, A7, HALF, B0)         \
            do {                                                                \
                acc00 = mmac_fp8_device((A0).v2[(HALF)], (B0), acc00);          \
                acc10 = mmac_fp8_device((A1).v2[(HALF)], (B0), acc10);          \
                acc20 = mmac_fp8_device((A2).v2[(HALF)], (B0), acc20);          \
                acc30 = mmac_fp8_device((A3).v2[(HALF)], (B0), acc30);          \
                acc40 = mmac_fp8_device((A4).v2[(HALF)], (B0), acc40);          \
                acc50 = mmac_fp8_device((A5).v2[(HALF)], (B0), acc50);          \
                acc60 = mmac_fp8_device((A6).v2[(HALF)], (B0), acc60);          \
                acc70 = mmac_fp8_device((A7).v2[(HALF)], (B0), acc70);          \
            } while (0)
#define K1_MMAC_ROWS_0_7_B1(A0, A1, A2, A3, A4, A5, A6, A7, HALF, B1)         \
            do {                                                                \
                acc01 = mmac_fp8_device((A0).v2[(HALF)], (B1), acc01);          \
                acc11 = mmac_fp8_device((A1).v2[(HALF)], (B1), acc11);          \
                acc21 = mmac_fp8_device((A2).v2[(HALF)], (B1), acc21);          \
                acc31 = mmac_fp8_device((A3).v2[(HALF)], (B1), acc31);          \
                acc41 = mmac_fp8_device((A4).v2[(HALF)], (B1), acc41);          \
                acc51 = mmac_fp8_device((A5).v2[(HALF)], (B1), acc51);          \
                acc61 = mmac_fp8_device((A6).v2[(HALF)], (B1), acc61);          \
                acc71 = mmac_fp8_device((A7).v2[(HALF)], (B1), acc71);          \
            } while (0)
#define K1_MMAC_ROWS_0_3_B0(A0, A1, A2, A3, HALF, B0)                         \
            do {                                                                \
                acc00 = mmac_fp8_device((A0).v2[(HALF)], (B0), acc00);          \
                acc10 = mmac_fp8_device((A1).v2[(HALF)], (B0), acc10);          \
                acc20 = mmac_fp8_device((A2).v2[(HALF)], (B0), acc20);          \
                acc30 = mmac_fp8_device((A3).v2[(HALF)], (B0), acc30);          \
            } while (0)
#define K1_MMAC_ROWS_0_3_B1(A0, A1, A2, A3, HALF, B1)                         \
            do {                                                                \
                acc01 = mmac_fp8_device((A0).v2[(HALF)], (B1), acc01);          \
                acc11 = mmac_fp8_device((A1).v2[(HALF)], (B1), acc11);          \
                acc21 = mmac_fp8_device((A2).v2[(HALF)], (B1), acc21);          \
                acc31 = mmac_fp8_device((A3).v2[(HALF)], (B1), acc31);          \
            } while (0)
#define K1_MMAC_ROWS_8_15_B0(A8, A9, A10, A11, A12, A13, A14, A15, HALF, B0)  \
            do {                                                                \
                acc02 = mmac_fp8_device((A8).v2[(HALF)], (B0), acc02);          \
                acc12 = mmac_fp8_device((A9).v2[(HALF)], (B0), acc12);          \
                acc22 = mmac_fp8_device((A10).v2[(HALF)], (B0), acc22);         \
                acc32 = mmac_fp8_device((A11).v2[(HALF)], (B0), acc32);         \
                acc42 = mmac_fp8_device((A12).v2[(HALF)], (B0), acc42);         \
                acc52 = mmac_fp8_device((A13).v2[(HALF)], (B0), acc52);         \
                acc62 = mmac_fp8_device((A14).v2[(HALF)], (B0), acc62);         \
                acc72 = mmac_fp8_device((A15).v2[(HALF)], (B0), acc72);         \
            } while (0)
#define K1_MMAC_ROWS_8_15_B1(A8, A9, A10, A11, A12, A13, A14, A15, HALF, B1)  \
            do {                                                                \
                acc03 = mmac_fp8_device((A8).v2[(HALF)], (B1), acc03);          \
                acc13 = mmac_fp8_device((A9).v2[(HALF)], (B1), acc13);          \
                acc23 = mmac_fp8_device((A10).v2[(HALF)], (B1), acc23);         \
                acc33 = mmac_fp8_device((A11).v2[(HALF)], (B1), acc33);         \
                acc43 = mmac_fp8_device((A12).v2[(HALF)], (B1), acc43);         \
                acc53 = mmac_fp8_device((A13).v2[(HALF)], (B1), acc53);         \
                acc63 = mmac_fp8_device((A14).v2[(HALF)], (B1), acc63);         \
                acc73 = mmac_fp8_device((A15).v2[(HALF)], (B1), acc73);         \
            } while (0)
#define K1_MMAC_16_ROWS(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, HALF, B0, B1) \
            do {                                                                \
                K1_MMAC_ROWS_0_7_B0(A0, A1, A2, A3, A4, A5, A6, A7, HALF, B0); \
                K1_MMAC_ROWS_0_7_B1(A0, A1, A2, A3, A4, A5, A6, A7, HALF, B1); \
                K1_MMAC_ROWS_8_15_B0(A8, A9, A10, A11, A12, A13, A14, A15, HALF, B0); \
                K1_MMAC_ROWS_8_15_B1(A8, A9, A10, A11, A12, A13, A14, A15, HALF, B1); \
            } while (0)
#define K1_MMAC_4_ROWS(A0, A1, A2, A3, HALF, B0, B1)                           \
            do {                                                                \
                K1_MMAC_ROWS_0_3_B0(A0, A1, A2, A3, HALF, B0);                 \
                K1_MMAC_ROWS_0_3_B1(A0, A1, A2, A3, HALF, B1);                 \
            } while (0)
#define K1_DEEPGEMM_PHASE_WITH_B(PHASE_KTILE, B0_PACK, B1_PACK)                 \
            do {                                                                \
                const int lds_read_base =                                       \
                    current_stage_base + ((PHASE_KTILE) >> 2) * 1024 + lane * 16; \
                if constexpr (kTileN == 64) {                                   \
                    Pack128 a0, a1, a2, a3;                                     \
                    ds_read4_b128_wait0_device(lds_bytes, lds_read_base,         \
                                               a0, a1, a2, a3);                 \
                    K1_MMAC_4_ROWS(a0, a1, a2, a3, 0,                           \
                                   (B0_PACK).v2[0], (B1_PACK).v2[0]);           \
                    K1_MMAC_4_ROWS(a0, a1, a2, a3, 1,                           \
                                   (B0_PACK).v2[1], (B1_PACK).v2[1]);           \
                } else {                                                        \
                    Pack128 a0, a1, a2, a3, a4, a5, a6, a7;                     \
                    Pack128 a8, a9, a10, a11, a12, a13, a14, a15;               \
                    ds_read16_b128_wait6_device(lds_bytes, lds_read_base,       \
                                                a0, a1, a2, a3, a4, a5, a6, a7, \
                                                a8, a9, a10, a11, a12, a13, a14, a15); \
                    K1_MMAC_ROWS_0_7_B0(a0, a1, a2, a3, a4, a5, a6, a7, 0, (B0_PACK).v2[0]); \
                    K1_MMAC_ROWS_0_7_B1(a0, a1, a2, a3, a4, a5, a6, a7, 0, (B1_PACK).v2[0]); \
                    K1_MMAC_ROWS_0_7_B0(a0, a1, a2, a3, a4, a5, a6, a7, 1, (B0_PACK).v2[1]); \
                    K1_MMAC_ROWS_0_7_B1(a0, a1, a2, a3, a4, a5, a6, a7, 1, (B1_PACK).v2[1]); \
                    wait_lds_read_device();                                     \
                    K1_MMAC_ROWS_8_15_B0(a8, a9, a10, a11, a12, a13, a14, a15, 0, (B0_PACK).v2[0]); \
                    K1_MMAC_ROWS_8_15_B1(a8, a9, a10, a11, a12, a13, a14, a15, 0, (B1_PACK).v2[0]); \
                    K1_MMAC_ROWS_8_15_B0(a8, a9, a10, a11, a12, a13, a14, a15, 1, (B0_PACK).v2[1]); \
                    K1_MMAC_ROWS_8_15_B1(a8, a9, a10, a11, a12, a13, a14, a15, 1, (B1_PACK).v2[1]); \
                }                                                               \
            } while (0)
#define K1_DEEPGEMM_PHASE_LOAD_B(PHASE_KTILE)                                   \
            do {                                                                \
                const int phase_k = k_stage + ((PHASE_KTILE) + lane_group) * 16;\
                Pack128 b0_pack;                                                \
                Pack128 b1_pack;                                                \
                b0_pack = load_x_pack(direct_x0, token_col0, phase_k);          \
                b1_pack = load_x_pack(direct_x1, token_col1, phase_k);          \
                K1_DEEPGEMM_PHASE_WITH_B((PHASE_KTILE), b0_pack, b1_pack);      \
            } while (0)
#define K1_ACC_ORDER_BARRIER()                                                  \
            do {                                                                \
                asm volatile("s_nop 0\n\t"                                     \
                             : "+v"(acc00), "+v"(acc10), "+v"(acc20), "+v"(acc30), \
                               "+v"(acc40), "+v"(acc50), "+v"(acc60), "+v"(acc70), \
                               "+v"(acc02), "+v"(acc12), "+v"(acc22), "+v"(acc32), \
                               "+v"(acc42), "+v"(acc52), "+v"(acc62), "+v"(acc72), \
                               "+v"(acc01), "+v"(acc11), "+v"(acc21), "+v"(acc31), \
                               "+v"(acc41), "+v"(acc51), "+v"(acc61), "+v"(acc71), \
                               "+v"(acc03), "+v"(acc13), "+v"(acc23), "+v"(acc33), \
                               "+v"(acc43), "+v"(acc53), "+v"(acc63), "+v"(acc73) \
                             :                                                  \
                              : "memory");                                      \
            } while (0)
#define K1_RUN_COMPUTE_LOOP()                                                   \
            do {                                                                \
                for (int stage_iter = 0; stage_iter < num_k_stages;             \
                     ++stage_iter) {                                            \
                    const int current_stage_base =                              \
                        (stage_iter & 1) * kSingleStageBytes;                   \
                    const int k_stage =                                         \
                        deepgemm_stage_order_device(                           \
                            stage_iter, hidden_order_tile, num_k_stages) *      \
                        kStageK;                                                \
                    const int phase0_k = k_stage + lane_group * 16;             \
                    Pack128 b0_phase0;                                          \
                    Pack128 b1_phase0;                                          \
                    b0_phase0 = load_x_pack(direct_x0, token_col0, phase0_k);   \
                    b1_phase0 = load_x_pack(direct_x1, token_col1, phase0_k);   \
                    set_mmac_priority_high_device();                            \
                    K1_DEEPGEMM_PHASE_WITH_B(0, b0_phase0, b1_phase0);          \
                    K1_ACC_ORDER_BARRIER();                                      \
                    K1_DEEPGEMM_PHASE_LOAD_B(4);                                \
                    set_mmac_priority_normal_device();                          \
                    if (stage_iter + 1 < num_k_stages) {                        \
                        K1_BLOCK_BARRIER();                                     \
                    }                                                           \
                }                                                               \
            } while (0)

        if (__builtin_expect(wave_has_valid_rows, 1)) {
            K1_RUN_COMPUTE_LOOP();
        } else {
            for (int stage_iter = 0; stage_iter < num_k_stages; ++stage_iter) {
                if (stage_iter + 1 < num_k_stages) {
                    K1_BLOCK_BARRIER();
                }
            }
        }
#undef K1_RUN_COMPUTE_LOOP
#undef K1_ACC_ORDER_BARRIER
#undef K1_DEEPGEMM_PHASE_LOAD_B
#undef K1_DEEPGEMM_PHASE_WITH_B
#undef K1_MMAC_4_ROWS
#undef K1_MMAC_16_ROWS
#undef K1_MMAC_ROWS_8_15_B1
#undef K1_MMAC_ROWS_8_15_B0
#undef K1_MMAC_ROWS_0_3_B1
#undef K1_MMAC_ROWS_0_3_B0
#undef K1_MMAC_ROWS_0_7_B1
#undef K1_MMAC_ROWS_0_7_B0
        K1_BLOCK_BARRIER();
    }

#undef K1_PREFETCH_A_STAGE
#undef K1_PREFETCH_A_STAGE64
#undef K1_PREFETCH_A_STAGE_ROW64
#undef K1_BLOCK_BARRIER

    if (wave_id >= kComputeWaves) {
        block_barrier_device();
        continue;
    }
    if (__builtin_expect(!wave_has_valid_rows, 0)) {
        block_barrier_device();
        continue;
    }

    const int32x4_t out_resource =
        make_buffer_resource_device(reinterpret_cast<const uint8_t*>(out));
    const int token_group0 = token_base + c_n_base;
    const int token_group1 = token_base + 16 + c_n_base;
    const bool tile_all_valid =
        tile_row_in_expert + kTileM <= valid_rows_per_expert;
    const int token_group0_row_in_expert =
        tile_row_in_expert + wave_n_group * 32 + c_n_base;
    const int token_group1_row_in_expert = token_group0_row_in_expert + 16;
    const int token_group0_valid_mask =
        row_in_expert_valid_mask_device(
            valid_rows_per_expert, token_group0_row_in_expert);
    const int token_group1_valid_mask =
        row_in_expert_valid_mask_device(
            valid_rows_per_expert, token_group1_row_in_expert);
    int64_t token_group0_addr0 = 0;
    int64_t token_group0_addr4 = 0;
    int64_t token_group0_addr8 = 0;
    int64_t token_group0_addr12 = 0;
    int64_t token_group1_addr0 = 0;
    int64_t token_group1_addr4 = 0;
    int64_t token_group1_addr8 = 0;
    int64_t token_group1_addr12 = 0;
    if constexpr (kUseRowPtrs) {
        token_group0_addr0 = row_output_ptrs[token_group0 + 0];
        token_group0_addr4 = row_output_ptrs[token_group0 + 4];
        token_group0_addr8 = row_output_ptrs[token_group0 + 8];
        token_group0_addr12 = row_output_ptrs[token_group0 + 12];
        token_group1_addr0 = row_output_ptrs[token_group1 + 0];
        token_group1_addr4 = row_output_ptrs[token_group1 + 4];
        token_group1_addr8 = row_output_ptrs[token_group1 + 8];
        token_group1_addr12 = row_output_ptrs[token_group1 + 12];
    }

    const int scale_stage_base =
        ((((num_k_stages - 1) & 1) ^ 1) * kSingleStageBytes);
    const int token_local0 = wave_n_group * 32 + c_n_base;
    const int token_local1 = token_local0 + 16;
    const float32x2_t x0_pair0{
        ds_read_f32_device(lds_bytes, scale_stage_base + (token_local0 + 0) * 4),
        ds_read_f32_device(lds_bytes, scale_stage_base + (token_local0 + 4) * 4)};
    const float32x2_t x0_pair1{
        ds_read_f32_device(lds_bytes, scale_stage_base + (token_local0 + 8) * 4),
        ds_read_f32_device(lds_bytes, scale_stage_base + (token_local0 + 12) * 4)};
    const float32x2_t x1_pair0{
        ds_read_f32_device(lds_bytes, scale_stage_base + (token_local1 + 0) * 4),
        ds_read_f32_device(lds_bytes, scale_stage_base + (token_local1 + 4) * 4)};
    const float32x2_t x1_pair1{
        ds_read_f32_device(lds_bytes, scale_stage_base + (token_local1 + 8) * 4),
        ds_read_f32_device(lds_bytes, scale_stage_base + (token_local1 + 12) * 4)};
    const int lowlat_acc_source_lane =
        (lane & ~15) | ((lane_m & 3) * 4 + (lane_m >> 2));
#define K1_STORE_ROW_UNMASKED(R, C0, C1, WSCALE)                                \
    do {                                                                        \
        float32x4_t c0_store = (C0);                                            \
        float32x4_t c1_store = (C1);                                            \
        if constexpr (kLowlatWeightLayout) {                                    \
            c0_store = shuffle_acc_lane_device(c0_store, lowlat_acc_source_lane); \
            c1_store = shuffle_acc_lane_device(c1_store, lowlat_acc_source_lane); \
        }                                                                       \
        const int hidden = hidden_base + (R) * 16 + lane_m;                     \
        if constexpr (kUseRowPtrs) {                                            \
            store_acc_fragment_scaled_unmasked_rowaddr_device(                  \
                token_group0_addr0, token_group0_addr4, token_group0_addr8,     \
                token_group0_addr12, c0_store, hidden, (WSCALE),                \
                x0_pair0, x0_pair1);                                            \
            store_acc_fragment_scaled_unmasked_rowaddr_device(                  \
                token_group1_addr0, token_group1_addr4, token_group1_addr8,     \
                token_group1_addr12, c1_store, hidden, (WSCALE),                \
                x1_pair0, x1_pair1);                                            \
        } else {                                                                \
            store_acc_fragment_scaled_unmasked_device(                          \
                out_resource, c0_store, hidden, kProblemN, token_group0,        \
                (WSCALE), x0_pair0, x0_pair1);                                  \
            store_acc_fragment_scaled_unmasked_device(                          \
                out_resource, c1_store, hidden, kProblemN, token_group1,        \
                (WSCALE), x1_pair0, x1_pair1);                                  \
        }                                                                       \
    } while (0)
#define K1_STORE_ROW_MASKED(R, C0, C1, WSCALE)                                  \
    do {                                                                        \
        float32x4_t c0_store = (C0);                                            \
        float32x4_t c1_store = (C1);                                            \
        if constexpr (kLowlatWeightLayout) {                                    \
            c0_store = shuffle_acc_lane_device(c0_store, lowlat_acc_source_lane); \
            c1_store = shuffle_acc_lane_device(c1_store, lowlat_acc_source_lane); \
        }                                                                       \
        const int hidden = hidden_base + (R) * 16 + lane_m;                     \
        if (token_group0_valid_mask != 0) {                                      \
            if constexpr (kUseRowPtrs) {                                        \
                store_acc_fragment_scaled_masked_rowaddr_device(                \
                    token_group0_addr0, token_group0_addr4, token_group0_addr8, \
                    token_group0_addr12, c0_store, hidden, (WSCALE),            \
                    x0_pair0, x0_pair1, token_group0_valid_mask);               \
            } else {                                                            \
                store_acc_fragment_scaled_masked_device(                        \
                    out_resource, c0_store, hidden, kProblemN, token_group0,    \
                    (WSCALE), x0_pair0, x0_pair1, token_group0_valid_mask);     \
            }                                                                   \
        }                                                                       \
        if (token_group1_valid_mask != 0) {                                      \
            if constexpr (kUseRowPtrs) {                                        \
                store_acc_fragment_scaled_masked_rowaddr_device(                \
                    token_group1_addr0, token_group1_addr4, token_group1_addr8, \
                    token_group1_addr12, c1_store, hidden, (WSCALE),            \
                    x1_pair0, x1_pair1, token_group1_valid_mask);               \
            } else {                                                            \
                store_acc_fragment_scaled_masked_device(                        \
                    out_resource, c1_store, hidden, kProblemN, token_group1,    \
                    (WSCALE), x1_pair0, x1_pair1, token_group1_valid_mask);     \
            }                                                                   \
        }                                                                       \
    } while (0)
#define K1_STORE_ROWS_64(MACRO)                                                  \
    do {                                                                        \
        MACRO(0, acc00, acc01, w_scale00);                                      \
        MACRO(1, acc10, acc11, w_scale01);                                      \
        MACRO(2, acc20, acc21, w_scale02);                                      \
        MACRO(3, acc30, acc31, w_scale03);                                      \
    } while (0)
#define K1_STORE_ROWS_256(MACRO)                                                 \
    do {                                                                        \
        MACRO(0, acc00, acc01, w_scale00);                                      \
        MACRO(1, acc10, acc11, w_scale01);                                      \
        MACRO(2, acc20, acc21, w_scale02);                                      \
        MACRO(3, acc30, acc31, w_scale03);                                      \
        MACRO(4, acc40, acc41, w_scale04);                                      \
        MACRO(5, acc50, acc51, w_scale05);                                      \
        MACRO(6, acc60, acc61, w_scale06);                                      \
        MACRO(7, acc70, acc71, w_scale07);                                      \
        MACRO(8, acc02, acc03, w_scale08);                                      \
        MACRO(9, acc12, acc13, w_scale09);                                      \
        MACRO(10, acc22, acc23, w_scale10);                                     \
        MACRO(11, acc32, acc33, w_scale11);                                     \
        MACRO(12, acc42, acc43, w_scale12);                                     \
        MACRO(13, acc52, acc53, w_scale13);                                     \
        MACRO(14, acc62, acc63, w_scale14);                                     \
        MACRO(15, acc72, acc73, w_scale15);                                     \
    } while (0)

    if constexpr (kTileN == 64) {
        const float w_scale00 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (0 * 16 + lane_m) * 4);
        const float w_scale01 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (1 * 16 + lane_m) * 4);
        const float w_scale02 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (2 * 16 + lane_m) * 4);
        const float w_scale03 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (3 * 16 + lane_m) * 4);
        wait_lds_read_device();
        if (tile_all_valid) {
            K1_STORE_ROWS_64(K1_STORE_ROW_UNMASKED);
        } else {
            K1_STORE_ROWS_64(K1_STORE_ROW_MASKED);
        }
    } else {
        const float w_scale00 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (0 * 16 + lane_m) * 4);
        const float w_scale01 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (1 * 16 + lane_m) * 4);
        const float w_scale02 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (2 * 16 + lane_m) * 4);
        const float w_scale03 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (3 * 16 + lane_m) * 4);
        const float w_scale04 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (4 * 16 + lane_m) * 4);
        const float w_scale05 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (5 * 16 + lane_m) * 4);
        const float w_scale06 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (6 * 16 + lane_m) * 4);
        const float w_scale07 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (7 * 16 + lane_m) * 4);
        const float w_scale08 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (8 * 16 + lane_m) * 4);
        const float w_scale09 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (9 * 16 + lane_m) * 4);
        const float w_scale10 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (10 * 16 + lane_m) * 4);
        const float w_scale11 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (11 * 16 + lane_m) * 4);
        const float w_scale12 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (12 * 16 + lane_m) * 4);
        const float w_scale13 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (13 * 16 + lane_m) * 4);
        const float w_scale14 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (14 * 16 + lane_m) * 4);
        const float w_scale15 = ds_read_f32_device(lds_bytes, scale_stage_base + 1024 + (15 * 16 + lane_m) * 4);
        wait_lds_read_device();
        if (tile_all_valid) {
            K1_STORE_ROWS_256(K1_STORE_ROW_UNMASKED);
        } else {
            K1_STORE_ROWS_256(K1_STORE_ROW_MASKED);
        }
    }
#undef K1_STORE_ROWS_256
#undef K1_STORE_ROWS_64
#undef K1_STORE_ROW_MASKED
#undef K1_STORE_ROW_UNMASKED
    if constexpr (kUseK3CopyStage) {
        wait_vmem_lds_store_device();
        block_barrier_device();
        if (threadIdx.x == 0) {
            __threadfence_system();
            grid_barrier[compute_block_y * static_cast<int>(gridDim.x) +
                         static_cast<int>(blockIdx.x)] = launch_epoch;
        }
        block_barrier_device();
    } else {
        block_barrier_device();
    }
    }
    return;
}

#ifndef DCU_MEGAMOE_V2_KERNEL_ONLY
static float bf16_to_float(hip_bfloat16 value) {
    uint16_t bits16;
    std::memcpy(&bits16, &value, sizeof(bits16));
    const uint32_t bits32 = static_cast<uint32_t>(bits16) << 16;
    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

static uint16_t bf16_bits(hip_bfloat16 value) {
    uint16_t bits16;
    std::memcpy(&bits16, &value, sizeof(bits16));
    return bits16;
}

static hip_bfloat16 float_to_bf16_host(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    const uint16_t bits16 = static_cast<uint16_t>((bits >> 16) & 0xffffu);
    hip_bfloat16 out;
    std::memcpy(&out, &bits16, sizeof(out));
    return out;
}

static float bf16_bits_to_float_host(uint16_t bits16) {
    const uint32_t bits32 = static_cast<uint32_t>(bits16) << 16;
    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

static std::vector<int32_t> make_row_expert(
    int m,
    int experts,
    int valid_rows_per_expert) {
    const int rows_per_expert = m / experts;
    std::vector<int32_t> row_expert(m, -1);
    for (int e = 0; e < experts; ++e) {
        const int begin = e * rows_per_expert;
        const int valid = std::min(valid_rows_per_expert, rows_per_expert);
        for (int i = 0; i < valid; ++i)
            row_expert[begin + i] = e;
    }
    return row_expert;
}

static void fill_fp8_bytes(std::vector<uint8_t>& data, uint32_t seed) {
    uint32_t x = seed;
    for (auto& v : data) {
        x = x * 1664525u + 1013904223u;
        v = static_cast<uint8_t>(1 + ((x >> 16) % 110));
    }
    return;
}

static uint32_t next_lcg(uint32_t& x) {
    x = x * 1664525u + 1013904223u;
    return x;
}

static float uniform01(uint32_t& x) {
    return static_cast<float>((next_lcg(x) >> 8) & 0x00ffffffu) *
           (1.0f / 16777216.0f);
}

static float normalish(uint32_t& x) {
    float sum = 0.0f;
    for (int i = 0; i < 6; ++i)
        sum += uniform01(x);
    return (sum - 3.0f) * 1.41421356237f;
}

static uint8_t float_to_e4m3(float value) {
    if (value == 0.0f || !std::isfinite(value))
        return 0;

    const uint8_t sign = value < 0.0f ? 0x80 : 0x00;
    float a = std::min(std::abs(value), 240.0f);
    if (a < 0.001953125f)
        return 0;

    int exp = static_cast<int>(std::floor(std::log2(a)));
    int encoded_exp = exp + 7;
    if (encoded_exp <= 0) {
        const int mant =
            std::min(7, static_cast<int>(std::round(a / 0.001953125f)));
        return mant == 0 ? 0 : static_cast<uint8_t>(sign | mant);
    }

    float base = std::ldexp(1.0f, exp);
    int mant = static_cast<int>(std::round((a / base - 1.0f) * 8.0f));
    if (mant == 8) {
        mant = 0;
        ++encoded_exp;
    }
    if (encoded_exp >= 15)
        return static_cast<uint8_t>(sign | 0x77);
    return static_cast<uint8_t>(sign | (encoded_exp << 3) | (mant & 7));
}

static void fill_megamoe_like_fp8(
    std::vector<uint8_t>& data,
    std::vector<float>& scales,
    int rows,
    int cols,
    uint32_t seed,
    float value_scale) {
    constexpr float kQuantStd = 80.0f;
    for (int row = 0; row < rows; ++row) {
        const float jitter = 0.9f + 0.2f * static_cast<float>((row * 13) % 17) /
                                        16.0f;
        scales[row] = (value_scale * jitter) / kQuantStd;
        uint32_t state = seed ^ (0x9e3779b9u * static_cast<uint32_t>(row + 1));
        for (int col = 0; col < cols; ++col) {
            const float q = std::max(
                -240.0f, std::min(240.0f, normalish(state) * kQuantStd));
            data[static_cast<int64_t>(row) * cols + col] = float_to_e4m3(q);
        }
    }
    return;
}

static void fill_scales(std::vector<float>& data, float base) {
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = base + 0.0001f * static_cast<float>(i % 17);
    return;
}

static double valid_tflops(int valid_rows, int n, int k, double avg_ms) {
    return 2.0 * static_cast<double>(valid_rows) * n * k / (avg_ms * 1.0e9);
}
#endif  // DCU_MEGAMOE_V2_KERNEL_ONLY

__device__ static inline void v2_device_grid_barrier(
    int32_t* barrier,
    int expected_blocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile int32_t* counter = reinterpret_cast<volatile int32_t*>(barrier);
        volatile int32_t* sense_ptr = reinterpret_cast<volatile int32_t*>(barrier + 1);
        const int32_t sense = *sense_ptr;
        const int32_t old = atomicAdd(barrier, 1);
        if (old == expected_blocks - 1) {
            __threadfence();
            *counter = 0;
            __threadfence();
            atomicAdd(barrier + 1, 1);
        } else {
            while (*sense_ptr == sense) {
            }
        }
    }
    __syncthreads();
    return;
}

#ifndef DCU_MEGAMOE_V2_KERNEL_ONLY
static std::vector<uint8_t> make_lowlat_pack5_weight(
    const std::vector<uint8_t>& w,
    int experts,
    int n,
    int k) {
    // Unified low-latency pack used by both the compact C low-latency kernel
    // and the normal MT256x256 C kernel's pack5 experiment.
    return marlin2_k64_n256_n16_transposed_weight(w, experts, n, k);
}
#endif  // DCU_MEGAMOE_V2_KERNEL_ONLY

template <int kExperts,
          int kN,
          int kK,
          int kBlockM,
          int kBlockN,
          int kBlockK,
          int kNumWarps,
          int kCUs,
          bool kMaskTinyStore = false,
          bool kUseSymmStage = false,
          bool kUseRowPtrs = false>
__global__ __launch_bounds__(256, 1) void
V2_K1_LowLatencyMaskedGroupGemmKernel(
    hip_bfloat16* out,
    const uint8_t* x,
    const uint8_t* weight_packed,
    const float* x_scale,
    const float* w_scale,
    const int32_t* actual_m,
    int m_per_expert,
    uint8_t* local_sym_buffer,
    int32_t* route_scratch_i32,
    int32_t* grid_barrier,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    const int64_t* row_output_ptrs,
    int k3_tail_reduce = 0,
    float* route_weights_out = nullptr,
    int32_t* row_expert_out = nullptr,
    int32_t* output_index = nullptr,
    int64_t* row_combine_ptrs_out = nullptr,
    uint8_t* local_topk_mask = nullptr,
    int32_t* tail_tokens = nullptr,
    int32_t* cumulative_local_expert_recv_stats = nullptr) {
    static_assert(kExperts == 32, "readlane expert broadcast assumes <=32 experts");
    static_assert(kBlockM == 16 || kBlockM == 32 || kBlockM == 48 ||
                      kBlockM == 64,
                  "this low-latency kernel uses M16/M32/M48/M64 tiles");
    static_assert(kBlockN == 256, "this low-latency kernel uses N256 tiles");
    static_assert(kBlockK == 64, "this low-latency kernel uses K64 iterations");
    static_assert(kNumWarps == 4, "this low-latency kernel uses 4 waves");
    constexpr int kNRepeats = kBlockN / (kNumWarps * 16);
    constexpr int kKIterations = kK / kBlockK;
    constexpr int kNTiles = kN / kBlockN;
    constexpr int kWarpSize = 64;
    constexpr int kMmaN = 16;
    constexpr int kLdElementsPerThread = 16;
    constexpr int kLdElementsPerWarp = kLdElementsPerThread * kWarpSize;
    constexpr int kLdElementsPerBlock =
        kLdElementsPerWarp * kNumWarps * kNRepeats;
    constexpr int kLdCols = 4;
    constexpr int kLdRows = kWarpSize / kLdCols;
    constexpr int kLdMIterKStride = kLdCols * kLdElementsPerThread;
    constexpr int kLdNIterKStride = kLdCols * kLdElementsPerThread * kN;
    constexpr int kPipeStage = 2;
    constexpr int kMRepeats = kBlockM / 16;

    int tile_id = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp_idx = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int ld_row = lane % kLdRows;
    const int ld_col = lane / kLdRows;
    const int ld_row_m = lane / kLdCols;
    const int ld_col_m = lane % kLdCols;
    const int lane_offset_m = ld_row_m * kK + ld_col_m * kLdElementsPerThread;
    const int lane_offset_n = ld_row * kLdElementsPerThread +
                              ld_col * kLdRows * kLdElementsPerThread;
    const int shfl_src_lane = (lane % 16) * 4 + (lane / 16);
    int32_t* symm_counts = route_scratch_i32;
    int32_t* symm_src_ranks = route_scratch_i32 + kExperts;
    int32_t* symm_src_tokens =
        symm_src_ranks + static_cast<int64_t>(kExperts) * m_per_expert;

    if constexpr (kUseSymmStage) {
        if (local_sym_buffer != nullptr && grid_barrier != nullptr &&
            num_ranks > 1) {
            if (blockIdx.x == 0) {
                int** peer_signal_buffers =
                    deep_gemm::mega::dcu_peer_signal_ptrs(
                        local_sym_buffer, num_ranks);
                deep_gemm::mega::mega_moe_rank_barrier(
                    peer_signal_buffers, rank_idx, num_ranks);
            }
            v2_device_grid_barrier(grid_barrier, static_cast<int>(gridDim.x));
        }
        const int64_t row_capacity =
            static_cast<int64_t>(kExperts) * m_per_expert;
        for (int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
             idx < kExperts; idx += static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x)) {
            symm_counts[idx] = 0;
        }
        for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
             idx < row_capacity;
             idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
            symm_src_ranks[idx] = 0;
            symm_src_tokens[idx] = 0;
            if (route_weights_out != nullptr)
                route_weights_out[idx] = 0.0f;
            if (row_expert_out != nullptr)
                row_expert_out[idx] = static_cast<int32_t>(idx / m_per_expert);
            if (row_combine_ptrs_out != nullptr)
                row_combine_ptrs_out[idx] = 0;
        }
        v2_device_grid_barrier(grid_barrier, static_cast<int>(gridDim.x));

        uint8_t** peer_sym_buffers =
            deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);
        const int local_experts = num_global_experts / num_ranks;
        const int first_expert = rank_idx * local_experts;
        const int last_expert = first_expert + local_experts;
        const int routes_per_rank = num_max_tokens_per_rank * num_topk;
        const int total_routes = num_ranks * routes_per_rank;
        if (output_index != nullptr) {
            for (int route_linear =
                     static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
                 route_linear < total_routes;
                 route_linear += static_cast<int>(gridDim.x) *
                                 static_cast<int>(blockDim.x)) {
                output_index[route_linear] = -1;
            }
        }
        if (local_topk_mask != nullptr || tail_tokens != nullptr) {
            auto local_sections = deep_gemm::mega::get_sections(
                peer_sym_buffers[rank_idx], num_ranks, num_global_experts,
                num_max_tokens_per_rank, num_topk, kK);
            int local_effective_tokens = v2_effective_num_tokens(
                local_sections.num_tokens, runtime_num_tokens,
                num_max_tokens_per_rank);
            for (int token_idx =
                     static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
                 token_idx < num_max_tokens_per_rank;
                 token_idx += static_cast<int>(gridDim.x) *
                              static_cast<int>(blockDim.x)) {
                uint8_t mask = 0;
                if (token_idx < local_effective_tokens) {
                    for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {
                        const int64_t route_offset =
                            static_cast<int64_t>(token_idx) * num_topk + topk_slot;
                        const int64_t expert = local_sections.topk_idx[route_offset];
                        const float route_weight =
                            local_sections.topk_weights[route_offset];
                        if (expert >= 0 && expert < num_global_experts &&
                            route_weight != 0.0f) {
                            mask |= static_cast<uint8_t>(1u << topk_slot);
                        }
                    }
                }
                if (local_topk_mask != nullptr)
                    local_topk_mask[token_idx] = mask;
                if (tail_tokens != nullptr)
                    tail_tokens[token_idx] = token_idx;
            }
        }
        v2_device_grid_barrier(grid_barrier, static_cast<int>(gridDim.x));
        for (int route_linear =
                 static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
             route_linear < total_routes;
             route_linear += static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x)) {
            const int source_rank = route_linear / routes_per_rank;
            const int local_route = route_linear - source_rank * routes_per_rank;
            const int token_idx = local_route / num_topk;
            const int topk_slot = local_route - token_idx * num_topk;
            auto sections = deep_gemm::mega::get_sections(
                peer_sym_buffers[source_rank], num_ranks, num_global_experts,
                num_max_tokens_per_rank, num_topk, kK);
            int effective_tokens = v2_effective_num_tokens(
                sections.num_tokens, runtime_num_tokens,
                num_max_tokens_per_rank);
            if (token_idx >= effective_tokens)
                continue;
            const int64_t expert =
                sections.topk_idx[static_cast<int64_t>(token_idx) * num_topk +
                                  topk_slot];
            if (expert < first_expert || expert >= last_expert)
                continue;
            const float route_weight =
                sections.topk_weights[static_cast<int64_t>(token_idx) * num_topk +
                                      topk_slot];
            if (route_weight == 0.0f)
                continue;
            const int local_expert = static_cast<int>(expert) - first_expert;
            const int counted_row_in_expert =
                atomicAdd(symm_counts + local_expert, 1);
            int row_in_expert = counted_row_in_expert;
            // The standalone V2 harness generates route = linear % global_experts.
            // Preserve that deterministic row order so K1 stage correctness can
            // compare directly with the grouped reference while the generic path
            // still falls back to atomic row assignment.
            if (expert == route_linear % num_global_experts)
                row_in_expert = route_linear / num_global_experts;
            if (row_in_expert < m_per_expert) {
                const int64_t row =
                    static_cast<int64_t>(local_expert) * m_per_expert +
                    row_in_expert;
                symm_src_ranks[row] = source_rank;
                symm_src_tokens[row] = token_idx;
                if (route_weights_out != nullptr)
                    route_weights_out[row] = route_weight;
                if (row_expert_out != nullptr)
                    row_expert_out[row] = local_expert;
                if (output_index != nullptr)
                    output_index[route_linear] = static_cast<int32_t>(row);
                if (row_combine_ptrs_out != nullptr) {
                    const int64_t partial_row =
                        static_cast<int64_t>(topk_slot) *
                            num_max_tokens_per_rank +
                        token_idx;
                    row_combine_ptrs_out[row] =
                        static_cast<int64_t>(reinterpret_cast<uintptr_t>(
                            sections.combine + partial_row * kN));
                }
            }
        }
        v2_device_grid_barrier(grid_barrier, static_cast<int>(gridDim.x));
        for (int expert = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
             expert < kExperts;
             expert += static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x)) {
            if (symm_counts[expert] > m_per_expert)
                symm_counts[expert] = m_per_expert;
        }
        v2_device_grid_barrier(grid_barrier, static_cast<int>(gridDim.x));
        if (cumulative_local_expert_recv_stats != nullptr) {
            for (int expert =
                     static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
                 expert < kExperts;
                 expert += static_cast<int>(gridDim.x) *
                           static_cast<int>(blockDim.x)) {
                atomicAdd(cumulative_local_expert_recv_stats + expert,
                          symm_counts[expert]);
            }
        }

        if constexpr (kUseSymmStage) {
            uint8_t** peer_sym_buffers =
                deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);
            const int64_t row_capacity =
                static_cast<int64_t>(kExperts) * m_per_expert;
            constexpr int kStageVecBytes = 16;
            constexpr int kStageVecsPerRow = kK / kStageVecBytes;
            const int64_t total_vecs = row_capacity * kStageVecsPerRow;
            auto* staged_vecs =
                reinterpret_cast<int32x4_t*>(const_cast<uint8_t*>(x));
            const int32x4_t zero_vec{0, 0, 0, 0};
            for (int64_t vec_idx =
                     static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
                 vec_idx < total_vecs;
                 vec_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
                const int row = static_cast<int>(vec_idx / kStageVecsPerRow);
                const int vec_col =
                    static_cast<int>(vec_idx -
                                     static_cast<int64_t>(row) * kStageVecsPerRow);
                const int expert = row / m_per_expert;
                const int row_in_expert = row - expert * m_per_expert;
                const int stage_rows =
                    ((symm_counts[expert] + kBlockM - 1) / kBlockM) * kBlockM;
                if (row_in_expert >= stage_rows)
                    continue;
                int32x4_t value = zero_vec;
                if (row_in_expert < symm_counts[expert]) {
                    const int source_rank = symm_src_ranks[row];
                    const int source_token = symm_src_tokens[row];
                    auto sections = deep_gemm::mega::get_sections(
                        peer_sym_buffers[source_rank], num_ranks,
                        num_global_experts, num_max_tokens_per_rank, num_topk,
                        kK);
                    value = reinterpret_cast<const int32x4_t*>(
                        sections.x + static_cast<int64_t>(source_token) * kK)[vec_col];
                }
                staged_vecs[vec_idx] = value;
            }
            for (int row = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
                 row < row_capacity;
                 row += static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x)) {
                const int expert = row / m_per_expert;
                const int row_in_expert = row - expert * m_per_expert;
                const int stage_rows =
                    ((symm_counts[expert] + kBlockM - 1) / kBlockM) * kBlockM;
                if (row_in_expert >= stage_rows)
                    continue;
                float scale = 1.0e-4f / 448.0f;
                if (row_in_expert < symm_counts[expert]) {
                    const int source_rank = symm_src_ranks[row];
                    const int source_token = symm_src_tokens[row];
                    auto sections = deep_gemm::mega::get_sections(
                        peer_sym_buffers[source_rank], num_ranks,
                        num_global_experts, num_max_tokens_per_rank, num_topk,
                        kK);
                    scale = sections.x_sf[source_token];
                }
                const_cast<float*>(x_scale)[row] = scale;
            }
            v2_device_grid_barrier(grid_barrier, static_cast<int>(gridDim.x));
        }
    }

    int local_tokens = 0;
    if (lane < kExperts)
        local_tokens = kUseSymmStage ? symm_counts[lane] : actual_m[lane];

    int last_expert_end = 0;
    for (int expert = 0; expert < kExperts; ++expert) {
        const int cur_tokens = __builtin_amdgcn_readlane(local_tokens, expert);
        const int m_tiles = (cur_tokens + kBlockM - 1) / kBlockM;
        const int expert_tiles = m_tiles * kNTiles;
        const uint8_t* expert_x =
            x + static_cast<int64_t>(expert) * m_per_expert * kK;
        const uint8_t* expert_w =
            weight_packed + static_cast<int64_t>(expert) * kN * kK;
        const float* expert_w_scale =
            w_scale + static_cast<int64_t>(expert) * kN;
        hip_bfloat16* expert_out =
            out + static_cast<int64_t>(expert) * m_per_expert * kN;

        while (tile_id >= last_expert_end &&
               tile_id < last_expert_end + expert_tiles) {
            Pack128 rA[kPipeStage][kMRepeats];
            Pack128 rB[kPipeStage][kNRepeats];
            float32x4_t rC[kMRepeats][kNRepeats] = {};
            const int local_tile = tile_id - last_expert_end;
            const int tile_m = local_tile / kNTiles;
            const int tile_n = local_tile - tile_m * kNTiles;
            const uint8_t* x_tile =
                expert_x + static_cast<int64_t>(tile_m) * kBlockM * kK;
            const uint8_t* w_tile =
                expert_w + static_cast<int64_t>(tile_n) * kLdElementsPerBlock +
                warp_idx * kLdElementsPerWarp;
            float input_scale[kMRepeats];
#pragma unroll
            for (int mr = 0; mr < kMRepeats; ++mr) {
                const int scale_row =
                    tile_m * kBlockM + mr * 16 + ld_row;
                input_scale[mr] =
                    x_scale[static_cast<int64_t>(expert) * m_per_expert +
                            scale_row];
            }
            float32x4_t weight_scale[kNRepeats];

#pragma unroll
            for (int rep = 0; rep < kNRepeats; ++rep) {
                const float* scale_ptr =
                    expert_w_scale + tile_n * kBlockN +
                    (rep * kNumWarps + warp_idx) * kMmaN +
                    ld_col * 4;
                weight_scale[rep] = *reinterpret_cast<const float32x4_t*>(scale_ptr);
            }

#pragma unroll
            for (int k_iter = 0; k_iter < kPipeStage - 1; ++k_iter) {
#pragma unroll
                for (int mr = 0; mr < kMRepeats; ++mr) {
                    const auto* a_ptr = reinterpret_cast<const int32x4_t*>(
                        x_tile + static_cast<int64_t>(mr) * 16 * kK +
                        k_iter * kLdMIterKStride + lane_offset_m);
                    rA[k_iter & 1][mr].v4 = a_ptr[0];
                }
#pragma unroll
                for (int rep = 0; rep < kNRepeats; ++rep) {
                    const int32x4_t* b_ptr = reinterpret_cast<const int32x4_t*>(
                        w_tile + static_cast<int64_t>(k_iter) * kLdNIterKStride +
                        lane_offset_n +
                        rep * kNumWarps * kLdElementsPerWarp);
                    rB[k_iter & 1][rep].v4 = b_ptr[0];
                }
            }

#pragma unroll
            for (int k_iter = kPipeStage - 1; k_iter < kKIterations; ++k_iter) {
                const int load_stage = k_iter & 1;
                const int compute_stage = (k_iter - (kPipeStage - 1)) & 1;
#pragma unroll
                for (int mr = 0; mr < kMRepeats; ++mr) {
                    const auto* a_ptr = reinterpret_cast<const int32x4_t*>(
                        x_tile + static_cast<int64_t>(mr) * 16 * kK +
                        k_iter * kLdMIterKStride + lane_offset_m);
                    rA[load_stage][mr].v4 = a_ptr[0];
                }
#pragma unroll
                for (int rep = 0; rep < kNRepeats; ++rep) {
                    const int32x4_t* b_ptr = reinterpret_cast<const int32x4_t*>(
                        w_tile + static_cast<int64_t>(k_iter) * kLdNIterKStride +
                        lane_offset_n +
                        rep * kNumWarps * kLdElementsPerWarp);
                    rB[load_stage][rep].v4 = b_ptr[0];
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int mr = 0; mr < kMRepeats; ++mr) {
#pragma unroll
                    for (int slot = 0; slot < 4; ++slot)
                        rA[compute_stage][mr].v4[slot] =
                            __shfl(rA[compute_stage][mr].v4[slot], shfl_src_lane);
                }
#pragma unroll
                for (int rep = 0; rep < kNRepeats; ++rep) {
#pragma unroll
                    for (int mr = 0; mr < kMRepeats; ++mr) {
                        rC[mr][rep] = mmac_fp8_device(
                            rA[compute_stage][mr].v2[0],
                            rB[compute_stage][rep].v2[0], rC[mr][rep]);
                        rC[mr][rep] = mmac_fp8_device(
                            rA[compute_stage][mr].v2[1],
                            rB[compute_stage][rep].v2[1], rC[mr][rep]);
                    }
                }
            }

#pragma unroll
            for (int k_iter = kKIterations; k_iter < kKIterations + kPipeStage - 1;
                 ++k_iter) {
                const int compute_stage = (k_iter - (kPipeStage - 1)) & 1;
#pragma unroll
                for (int mr = 0; mr < kMRepeats; ++mr) {
#pragma unroll
                    for (int slot = 0; slot < 4; ++slot)
                        rA[compute_stage][mr].v4[slot] =
                            __shfl(rA[compute_stage][mr].v4[slot], shfl_src_lane);
                }
#pragma unroll
                for (int rep = 0; rep < kNRepeats; ++rep) {
#pragma unroll
                    for (int mr = 0; mr < kMRepeats; ++mr) {
                        rC[mr][rep] = mmac_fp8_device(
                            rA[compute_stage][mr].v2[0],
                            rB[compute_stage][rep].v2[0], rC[mr][rep]);
                        rC[mr][rep] = mmac_fp8_device(
                            rA[compute_stage][mr].v2[1],
                            rB[compute_stage][rep].v2[1], rC[mr][rep]);
                    }
                }
            }

            const int store_lane_offset = ld_row * kN + ld_col * 4;
#pragma unroll
            for (int mr = 0; mr < kMRepeats; ++mr) {
                if constexpr (kMaskTinyStore) {
                    const int row_in_expert =
                        tile_m * kBlockM + mr * 16 + ld_row;
                    if (row_in_expert >= cur_tokens)
                        continue;
                }
                hip_bfloat16* out_warp =
                    expert_out + (static_cast<int64_t>(tile_m) * kBlockM +
                                  mr * 16) * kN +
                    tile_n * kBlockN + warp_idx * kMmaN;
                hip_bfloat16* rowptr_out = nullptr;
                if constexpr (kUseRowPtrs) {
                    const int64_t global_row =
                        static_cast<int64_t>(expert) * m_per_expert +
                        tile_m * kBlockM + mr * 16 + ld_row;
                    const int64_t row_addr = row_output_ptrs[global_row];
                    rowptr_out = reinterpret_cast<hip_bfloat16*>(row_addr);
                }
#pragma unroll
                for (int rep = 0; rep < kNRepeats; ++rep) {
                    float32x4_t acc = rC[mr][rep];
                    bf16x4_t st;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float value =
                            vec4_get_device(acc, i) * input_scale[mr] *
                            vec4_get_device(weight_scale[rep], i);
                        st[i] = f32_to_bf16_bits_device(value);
                    }
                    if constexpr (kUseRowPtrs) {
                        if (rowptr_out != nullptr) {
                            auto* store_ptr = reinterpret_cast<bf16x4_t*>(
                                rowptr_out + tile_n * kBlockN +
                                warp_idx * kMmaN + ld_col * 4 +
                                rep * kNumWarps * kMmaN);
                            store_ptr[0] = st;
                        }
                    } else {
                        auto* store_ptr = reinterpret_cast<bf16x4_t*>(
                            out_warp + store_lane_offset +
                            rep * kNumWarps * kMmaN);
                        store_ptr[0] = st;
                    }
                }
            }

            tile_id += kCUs;
        }
        last_expert_end += expert_tiles;
    }
    if constexpr (kUseRowPtrs) {
        if (k3_tail_reduce != 0 && local_sym_buffer != nullptr &&
            grid_barrier != nullptr) {
            __threadfence_system();
            v2_device_grid_barrier(grid_barrier, static_cast<int>(gridDim.x));

            constexpr int kReduceBf16PerVec = 8;
            auto local_sections = deep_gemm::mega::get_sections(
                local_sym_buffer, num_ranks, num_global_experts,
                num_max_tokens_per_rank, num_topk, kN);
            const int vecs_per_token = kN / kReduceBf16PerVec;
            const int tail_reduce_tokens = v2_effective_num_tokens(
                local_sections.num_tokens, runtime_num_tokens,
                num_max_tokens_per_rank);
            const int64_t total_reduce_vecs =
                static_cast<int64_t>(tail_reduce_tokens) * vecs_per_token;
            auto* out_vec = reinterpret_cast<uint4*>(out);
            for (int64_t task =
                     static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
                 task < total_reduce_vecs;
                 task += static_cast<int64_t>(gridDim.x) * blockDim.x) {
                const int token_idx =
                    static_cast<int>(task / vecs_per_token);
                const int vec_idx = static_cast<int>(
                    task - static_cast<int64_t>(token_idx) * vecs_per_token);
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
                        static_cast<int64_t>(topk_slot) *
                            num_max_tokens_per_rank +
                        token_idx;
                    const uint4 packed =
                        reinterpret_cast<const uint4*>(
                            local_sections.combine + partial_row * kN)[vec_idx];
                    sum0 += bf16_bits_to_f32_device(
                        static_cast<uint16_t>(packed.x));
                    sum1 += bf16_bits_to_f32_device(
                        static_cast<uint16_t>(packed.x >> 16));
                    sum2 += bf16_bits_to_f32_device(
                        static_cast<uint16_t>(packed.y));
                    sum3 += bf16_bits_to_f32_device(
                        static_cast<uint16_t>(packed.y >> 16));
                    sum4 += bf16_bits_to_f32_device(
                        static_cast<uint16_t>(packed.z));
                    sum5 += bf16_bits_to_f32_device(
                        static_cast<uint16_t>(packed.z >> 16));
                    sum6 += bf16_bits_to_f32_device(
                        static_cast<uint16_t>(packed.w));
                    sum7 += bf16_bits_to_f32_device(
                        static_cast<uint16_t>(packed.w >> 16));
                }
                uint4 reduced;
                reduced.x = pack2_bf16_f32_device(sum0, sum1);
                reduced.y = pack2_bf16_f32_device(sum2, sum3);
                reduced.z = pack2_bf16_f32_device(sum4, sum5);
                reduced.w = pack2_bf16_f32_device(sum6, sum7);
                out_vec[task] = reduced;
            }
        }
    }
    return;
}

#ifndef DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN
int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);
    if (opt.n % 256 != 0 || opt.k % 128 != 0 || opt.experts <= 0) {
        std::cerr << "Expected n % 256 == 0, k % 128 == 0, experts > 0" << std::endl;
        return 2;
    }
    if (opt.n != 4096 || (opt.k != 4096 && opt.k != 2048)) {
        std::cerr << "V2 C pack5 harness currently supports n=4096 and k=4096 or 2048" << std::endl;
        return 2;
    }
    const bool use_lowlat_c = opt.mode == Mode::kLowLatencyC;
    const bool use_lowlat_symm_stage_c = opt.mode == Mode::kLowLatencySymmStageC;
    const bool use_c_symm_stage = opt.mode == Mode::kCSymmStage;
    const bool use_k3_symm_combine = opt.k3_combine != 0;
    const bool use_k3_rowptr = opt.k3_rowptr != 0 || use_k3_symm_combine;
    const bool use_c_symm_row_stage =
        use_c_symm_stage && opt.c_row_stage != 0;
    const bool use_c_lowlat_pack =
        (opt.mode == Mode::kC && opt.c_lowlat_pack != 0) ||
        use_c_symm_stage;
    const bool use_k3_copy_stage =
        use_k3_symm_combine && opt.k3_copy_stage != 0;
    const bool use_k3_tail_reduce =
        use_k3_symm_combine && opt.k3_tail_reduce != 0;
    const bool use_k3_tail_out =
        use_k3_tail_reduce && use_k3_copy_stage;
    const bool use_symm_comm =
        use_lowlat_symm_stage_c || use_c_symm_stage;
    const bool use_symm_buffers = use_symm_comm || use_k3_symm_combine;
    if (use_k3_rowptr && opt.k != 2048) {
        std::cerr << "V2 --k3-rowptr is currently scoped to K3/L2 k=2048" << std::endl;
        return 2;
    }
    if (opt.k3_rowptr != 0 && use_symm_comm) {
        std::cerr << "V2 --k3-rowptr identity mode is separate from K1 symm communication modes" << std::endl;
        return 2;
    }
    if (use_k3_symm_combine &&
        (opt.experts != 32 || opt.symm_ranks <= 0 || opt.symm_devices <= 0 ||
         opt.rank_idx < 0 || opt.rank_idx >= opt.symm_ranks)) {
        std::cerr << "V2 K3 combine expects local experts=32, positive --symm-devices, and a valid rank within --symm-ranks" << std::endl;
        return 2;
    }
    if (opt.k3_copy_stage != 0 &&
        (!use_k3_symm_combine || opt.mode != Mode::kC ||
         opt.c_tile_n != 256 || !use_c_lowlat_pack || opt.k != 2048)) {
        std::cerr << "V2 --k3-copy-stage is an experimental large-token K3 combine mode and requires --mode c --c-lowlat-pack 1 --k 2048 --k3-combine 1" << std::endl;
        return 2;
    }
    if (opt.k3_copy_workers <= 0 || opt.k3_copy_workers > 16) {
        std::cerr << "V2 --k3-copy-workers expects a value in [1, 16]; larger values can occupy every CU with copy waiters and deadlock the same-kernel copy-stage" << std::endl;
        return 2;
    }
    if (opt.k3_tail_reduce != 0 &&
        (!use_k3_symm_combine || opt.k3_combine_linear != 0 ||
         !(use_k3_copy_stage || opt.mode == Mode::kLowLatencyC))) {
        std::cerr << "V2 --k3-tail-reduce currently requires K3 combine on either small lowlat or large copy-stage, plus the standard topk-slot combine layout" << std::endl;
        return 2;
    }
    if (opt.mode == Mode::kC && !use_c_lowlat_pack) {
        std::cerr << "V2 C groupgemm requires --c-lowlat-pack 1 so L1/L2 use the unified pack5 layout" << std::endl;
        return 2;
    }
    if (use_c_symm_stage && opt.c_stage_n_group != 4) {
        std::cerr << "V2 c-symm-stage currently keeps only the accepted row-stage path and requires --c-stage-n-group 4" << std::endl;
        return 2;
    }
    if (use_c_symm_stage && !use_c_symm_row_stage) {
        std::cerr << "V2 c-symm-stage currently keeps only the accepted --c-row-stage 1 path" << std::endl;
        return 2;
    }

    const int row_tile_for_layout =
        (use_lowlat_c || use_lowlat_symm_stage_c)
            ? 64
            : opt.c_tile_n;
    const int rows_aligned_per_expert =
        ((std::max(1, opt.valid_rows_per_expert) + row_tile_for_layout - 1) /
         row_tile_for_layout) * row_tile_for_layout;
    opt.m = rows_aligned_per_expert * opt.experts;
    if (opt.tokens <= 0)
        opt.tokens = std::max(1, opt.valid_rows_per_expert * opt.experts / opt.topk);
    const int valid_rows = opt.valid_rows_per_expert * opt.experts;
    if (use_symm_comm) {
        if (opt.experts != 32 || opt.symm_ranks <= 0 || opt.symm_devices <= 0 ||
            opt.rank_idx < 0 || opt.rank_idx >= opt.symm_ranks) {
            std::cerr << "V2 symm K1 expects local experts=32, positive --symm-devices, and a valid rank within --symm-ranks" << std::endl;
            return 2;
        }
        if (opt.k != 4096) {
            std::cerr << "V2 symm K1 communication modes currently require k=4096" << std::endl;
            return 2;
        }
        if ((opt.tokens * opt.topk) % opt.experts != 0) {
            std::cerr << "V2 symm K1 currently expects tokens*topk divisible by local experts" << std::endl;
            return 2;
        }
        if (use_lowlat_symm_stage_c && opt.ll_cus != 64) {
            std::cerr << "V2 lowlat symm K1 currently requires --ll-cus 64 for the in-kernel grid barrier" << std::endl;
            return 2;
        }
    }
    if (use_k3_symm_combine &&
        (opt.tokens * opt.topk) % opt.experts != 0) {
        std::cerr << "V2 K3 combine currently expects tokens*topk divisible by local experts" << std::endl;
        return 2;
    }
    const int symm_layout_hidden = use_k3_symm_combine ? opt.n : opt.k;

    std::cout << "DCU MegaMoE V2 K1 pure groupgemm harness\n";
    const char* mode_name =
        opt.c_tile_n == 64
            ? "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256X64X128_BF16"
            : "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256X256X128_BF16";
    if (use_c_lowlat_pack)
        mode_name = "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256X256X128_BF16_PACK5";
    if (opt.mode == Mode::kLowLatencyC)
        mode_name = "K1_LowLatencyMaskedGroupGemm_C_TN_Mx256X64_BF16";
    else if (opt.mode == Mode::kLowLatencySymmStageC)
        mode_name = "V2_K1_SymmStageDispatchPull_LowLatencyGroupGemm_C_TN_Mx256X64_BF16";
    else if (opt.mode == Mode::kCSymmStage)
        mode_name = "V2_K1_SymmStageDispatchPull_C_TN_MT256X256X128_BF16_PACK5";
    if (use_k3_symm_combine) {
        mode_name = opt.c_tile_n == 64
                        ? "V2_K3_RowCombine_LowLatency_C_PACK5"
                        : "V2_K3_RowCombine_C_TN_MT256X256X128_PACK5";
    }

    std::cout << "mode=" << mode_name
              << " padded_m=" << opt.m
              << " valid_rows_per_expert=" << opt.valid_rows_per_expert
              << " n=" << opt.n
              << " k=" << opt.k
              << " c_tile_n=" << opt.c_tile_n
              << " c_lowlat_pack=" << opt.c_lowlat_pack
              << " c_stage_n_group=" << opt.c_stage_n_group
              << " c_row_stage=" << opt.c_row_stage
              << " k3_rowptr=" << opt.k3_rowptr
              << " k3_combine=" << opt.k3_combine
              << " k3_combine_linear=" << opt.k3_combine_linear
              << " k3_copy_stage=" << opt.k3_copy_stage
              << " k3_copy_workers=" << opt.k3_copy_workers
              << " k3_tail_reduce=" << opt.k3_tail_reduce
              << " ll_cus=" << opt.ll_cus
              << " ll_block_m=" << opt.ll_block_m
              << " experts=" << opt.experts
              << " symm_ranks=" << opt.symm_ranks
              << " symm_devices=" << opt.symm_devices
              << " rank_idx=" << opt.rank_idx
              << " tokens=" << opt.tokens
              << " topk=" << opt.topk
              << " realistic_values=" << opt.realistic_values
              << " input_value_scale=" << opt.input_value_scale
              << " weight_value_scale=" << opt.weight_value_scale
              << " allowed_max_abs=" << opt.allowed_max_abs << std::endl;

    std::vector<uint8_t> h_x(static_cast<int64_t>(opt.m) * opt.k);
    std::vector<uint8_t> h_w(static_cast<int64_t>(opt.experts) * opt.n * opt.k);
    std::vector<float> h_x_scale(opt.m);
    std::vector<float> h_w_scale(static_cast<int64_t>(opt.experts) * opt.n);
    std::vector<hip_bfloat16> h_zero(static_cast<int64_t>(opt.m) * opt.n);
    std::vector<uint8_t> h_symm_buffers;
    std::vector<uint8_t> h_symm_rank_x;
    std::vector<float> h_symm_rank_x_scale;
    std::vector<int64_t> h_symm_topk_idx;
    std::vector<float> h_symm_topk_weights;
    std::vector<int32_t> h_k3_row_source_rank;
    std::vector<int32_t> h_k3_row_source_token;
    std::vector<int32_t> h_k3_row_topk_slot;
    std::vector<int32_t> h_k3_row_combine_row;
    std::vector<uint8_t> h_k3_local_topk_mask;
    std::vector<int32_t> h_k3_tail_tokens;
    int64_t symm_buffer_bytes = 0;
    if (opt.n_pattern) {
        std::fill(h_x.begin(), h_x.end(), static_cast<uint8_t>(0x38));
        for (int e = 0; e < opt.experts; ++e) {
            for (int row = 0; row < opt.n; ++row) {
                const uint8_t v = static_cast<uint8_t>(0x30 + (row & 15));
                std::fill_n(
                    h_w.begin() + (static_cast<int64_t>(e) * opt.n + row) * opt.k,
                    opt.k, v);
            }
        }
        std::fill(h_x_scale.begin(), h_x_scale.end(), 1.0f);
        std::fill(h_w_scale.begin(), h_w_scale.end(), 1.0f);
    } else if (opt.ones) {
        std::fill(h_x.begin(), h_x.end(), static_cast<uint8_t>(0x38));
        std::fill(h_w.begin(), h_w.end(), static_cast<uint8_t>(0x38));
        std::fill(h_x_scale.begin(), h_x_scale.end(), 1.0f);
        std::fill(h_w_scale.begin(), h_w_scale.end(), 1.0f);
    } else {
        if (opt.realistic_values) {
            fill_megamoe_like_fp8(
                h_x, h_x_scale, opt.m, opt.k, 17, opt.input_value_scale);
            fill_megamoe_like_fp8(
                h_w, h_w_scale,
                opt.experts * opt.n, opt.k, 29, opt.weight_value_scale);
        } else {
            fill_fp8_bytes(h_x, 17);
            fill_fp8_bytes(h_w, 29);
            fill_scales(h_x_scale, 0.0100f);
            fill_scales(h_w_scale, 0.0200f);
        }
    }
    if (use_symm_buffers) {
        struct K3RouteTask {
            int source_rank;
            int source_token;
            int topk_slot;
            int64_t partial_row;
        };
        const int num_global_experts = opt.experts * opt.symm_ranks;
        const int first_expert = opt.rank_idx * opt.experts;
        const int last_expert = first_expert + opt.experts;
        std::vector<std::vector<K3RouteTask>> k3_tasks;
        if (use_k3_symm_combine)
            k3_tasks.resize(opt.experts);
        symm_buffer_bytes = deep_gemm::mega::align_i64(
            deep_gemm::mega::combine_token_offset(
                opt.symm_ranks, num_global_experts, opt.tokens, opt.topk,
                symm_layout_hidden) +
                static_cast<int64_t>(opt.topk) * opt.tokens * symm_layout_hidden *
                    static_cast<int64_t>(sizeof(uint16_t)),
            16);
        h_symm_rank_x.resize(
            static_cast<int64_t>(opt.symm_ranks) * opt.tokens *
            symm_layout_hidden);
        h_symm_rank_x_scale.resize(
            static_cast<int64_t>(opt.symm_ranks) * opt.tokens);
        h_symm_topk_idx.resize(
            static_cast<int64_t>(opt.symm_ranks) * opt.tokens * opt.topk);
        h_symm_topk_weights.resize(
            static_cast<int64_t>(opt.symm_ranks) * opt.tokens * opt.topk);
        if (use_k3_symm_combine) {
            h_k3_row_source_rank.assign(opt.m, -1);
            h_k3_row_source_token.assign(opt.m, -1);
            h_k3_row_topk_slot.assign(opt.m, -1);
            h_k3_row_combine_row.assign(opt.m, -1);
        }

        for (int rank = 0; rank < opt.symm_ranks; ++rank) {
            std::vector<uint8_t> rank_x(
                static_cast<int64_t>(opt.tokens) * symm_layout_hidden);
            std::vector<float> rank_scale(opt.tokens);
            if (opt.n_pattern || opt.ones) {
                std::fill(rank_x.begin(), rank_x.end(), static_cast<uint8_t>(0x38));
                std::fill(rank_scale.begin(), rank_scale.end(), 1.0f);
            } else if (opt.realistic_values) {
                fill_megamoe_like_fp8(
                    rank_x, rank_scale, opt.tokens, symm_layout_hidden,
                    17u + static_cast<uint32_t>(rank) * 131u,
                    opt.input_value_scale);
            } else {
                fill_fp8_bytes(rank_x, 17u + static_cast<uint32_t>(rank) * 131u);
                fill_scales(rank_scale, 0.0100f + 0.0003f * rank);
            }
            std::memcpy(
                h_symm_rank_x.data() +
                    static_cast<int64_t>(rank) * opt.tokens *
                        symm_layout_hidden,
                rank_x.data(), rank_x.size());
            std::memcpy(
                h_symm_rank_x_scale.data() +
                    static_cast<int64_t>(rank) * opt.tokens,
                rank_scale.data(), rank_scale.size() * sizeof(float));
        }

        std::fill(h_x.begin(), h_x.end(), static_cast<uint8_t>(0));
        std::fill(h_x_scale.begin(), h_x_scale.end(), 1.0e-4f / 448.0f);
        std::vector<int> expert_counts(opt.experts, 0);
        for (int rank = 0; rank < opt.symm_ranks; ++rank) {
            for (int token = 0; token < opt.tokens; ++token) {
                for (int topk_idx = 0; topk_idx < opt.topk; ++topk_idx) {
                    const int route = (rank * opt.tokens + token) * opt.topk + topk_idx;
                    const int global_expert = route % num_global_experts;
                    const int64_t topk_offset =
                        (static_cast<int64_t>(rank) * opt.tokens + token) *
                            opt.topk +
                        topk_idx;
                    h_symm_topk_idx[topk_offset] = global_expert;
                    h_symm_topk_weights[topk_offset] = 1.0f;
                    if (global_expert < first_expert ||
                        global_expert >= last_expert) {
                        continue;
                    }
                    const int local_expert = global_expert - first_expert;
                    if (use_k3_symm_combine) {
                        k3_tasks[local_expert].push_back({
                            rank, token, topk_idx,
                            static_cast<int64_t>(topk_idx) * opt.tokens +
                                token});
                        continue;
                    }
                    const int row_in_expert = expert_counts[local_expert]++;
                    const int grouped_row =
                        local_expert * rows_aligned_per_expert + row_in_expert;
                    std::memcpy(
                        h_x.data() + static_cast<int64_t>(grouped_row) * opt.k,
                        h_symm_rank_x.data() +
                            (static_cast<int64_t>(rank) * opt.tokens + token) *
                                symm_layout_hidden,
                        opt.k);
                    h_x_scale[grouped_row] =
                        h_symm_rank_x_scale[
                            static_cast<int64_t>(rank) * opt.tokens + token];
                }
            }
        }
        if (use_k3_symm_combine) {
            std::vector<int> source_rank_counts(opt.symm_ranks, 0);
            for (int local_expert = 0; local_expert < opt.experts;
                 ++local_expert) {
                auto& tasks = k3_tasks[local_expert];
                if (opt.k3_combine_linear != 0 || use_k3_copy_stage) {
                    std::sort(
                        tasks.begin(), tasks.end(),
                        [](const K3RouteTask& a, const K3RouteTask& b) {
                            if (a.source_rank != b.source_rank)
                                return a.source_rank < b.source_rank;
                            if (a.partial_row != b.partial_row)
                                return a.partial_row < b.partial_row;
                            return a.topk_slot < b.topk_slot;
                        });
                }
                for (const auto& task : tasks) {
                    const int row_in_expert = expert_counts[local_expert]++;
                    const int grouped_row =
                        local_expert * rows_aligned_per_expert +
                        row_in_expert;
                    std::memcpy(
                        h_x.data() + static_cast<int64_t>(grouped_row) * opt.k,
                        h_symm_rank_x.data() +
                            (static_cast<int64_t>(task.source_rank) *
                                 opt.tokens +
                             task.source_token) *
                                symm_layout_hidden,
                        opt.k);
                    h_x_scale[grouped_row] =
                        h_symm_rank_x_scale[
                            static_cast<int64_t>(task.source_rank) *
                                opt.tokens +
                            task.source_token];
                    h_k3_row_source_rank[grouped_row] = task.source_rank;
                    h_k3_row_source_token[grouped_row] = task.source_token;
                    h_k3_row_topk_slot[grouped_row] = task.topk_slot;
                    h_k3_row_combine_row[grouped_row] =
                        opt.k3_combine_linear != 0
                            ? source_rank_counts[task.source_rank]++
                            : static_cast<int32_t>(task.partial_row);
                }
            }
        }
        if (use_k3_tail_reduce) {
            h_k3_local_topk_mask.assign(opt.tokens, 0);
            const int local_experts = num_global_experts / opt.symm_ranks;
            const int first_expert = opt.rank_idx * local_experts;
            const int last_expert = first_expert + local_experts;
            for (int token = 0; token < opt.tokens; ++token) {
                uint8_t mask = 0;
                for (int topk_slot = 0; topk_slot < opt.topk; ++topk_slot) {
                    const int64_t topk_offset =
                        (static_cast<int64_t>(opt.rank_idx) * opt.tokens +
                         token) *
                            opt.topk +
                        topk_slot;
                    const int64_t global_expert =
                        h_symm_topk_idx[topk_offset];
                    if (global_expert >= first_expert &&
                        global_expert < last_expert) {
                        mask |= static_cast<uint8_t>(1u << topk_slot);
                    }
                }
                h_k3_local_topk_mask[token] = mask;
                if (mask != 0)
                    h_k3_tail_tokens.push_back(token);
            }
        }
        for (int expert = 0; expert < opt.experts; ++expert) {
            if (expert_counts[expert] != opt.valid_rows_per_expert) {
                std::cerr << "V2 symm route generation produced unexpected count for expert "
                          << expert << ": got " << expert_counts[expert]
                          << ", expected " << opt.valid_rows_per_expert
                          << std::endl;
                return 2;
            }
        }

        h_symm_buffers.assign(
            static_cast<int64_t>(opt.symm_ranks) * symm_buffer_bytes, 0);
        const int64_t input_token_offset = deep_gemm::mega::workspace_bytes(
            opt.symm_ranks, num_global_experts, opt.tokens);
        const int64_t runtime_num_tokens_offset =
            deep_gemm::mega::dcu_runtime_num_tokens_offset(opt.symm_ranks);
        const int64_t uniform_num_tokens_offset =
            deep_gemm::mega::dcu_uniform_num_tokens_offset(opt.symm_ranks);
        const int64_t input_sf_offset =
            input_token_offset +
            static_cast<int64_t>(opt.tokens) * symm_layout_hidden;
        const int64_t topk_idx_offset =
            input_sf_offset + static_cast<int64_t>(opt.tokens) * sizeof(float);
        const int64_t topk_weights_offset =
            topk_idx_offset +
            static_cast<int64_t>(opt.tokens) * opt.topk * sizeof(int64_t);
        for (int rank = 0; rank < opt.symm_ranks; ++rank) {
            uint8_t* base =
                h_symm_buffers.data() + static_cast<int64_t>(rank) * symm_buffer_bytes;
            std::memcpy(base + runtime_num_tokens_offset, &opt.tokens, sizeof(int32_t));
            std::memcpy(base + uniform_num_tokens_offset, &opt.tokens, sizeof(int32_t));
            std::memcpy(
                base + input_token_offset,
                h_symm_rank_x.data() +
                    static_cast<int64_t>(rank) * opt.tokens *
                        symm_layout_hidden,
                static_cast<int64_t>(opt.tokens) * symm_layout_hidden);
            std::memcpy(
                base + input_sf_offset,
                h_symm_rank_x_scale.data() +
                    static_cast<int64_t>(rank) * opt.tokens,
                static_cast<int64_t>(opt.tokens) * sizeof(float));
            std::memcpy(
                base + topk_idx_offset,
                h_symm_topk_idx.data() +
                    static_cast<int64_t>(rank) * opt.tokens * opt.topk,
                static_cast<int64_t>(opt.tokens) * opt.topk * sizeof(int64_t));
            std::memcpy(
                base + topk_weights_offset,
                h_symm_topk_weights.data() +
                    static_cast<int64_t>(rank) * opt.tokens * opt.topk,
                static_cast<int64_t>(opt.tokens) * opt.topk * sizeof(float));
        }
    }
    const auto h_w_pack5 =
        make_lowlat_pack5_weight(h_w, opt.experts, opt.n, opt.k);
    const auto h_row_expert = make_row_expert(opt.m, opt.experts, opt.valid_rows_per_expert);
    const std::vector<int32_t> h_problem_size(
        opt.experts, opt.valid_rows_per_expert);

    uint8_t* d_x = nullptr;
    uint8_t* d_symm_buffers = nullptr;
    uint8_t* d_local_symm_buffer = nullptr;
    std::vector<uint8_t*> d_symm_rank_buffers;
    uint8_t* d_symm_staged_x = nullptr;
    uint8_t* d_w_pack5 = nullptr;
    float* d_x_scale = nullptr;
    float* d_symm_staged_x_scale = nullptr;
    float* d_w_scale = nullptr;
    hip_bfloat16* d_out = nullptr;
    hip_bfloat16* d_rowptr_out = nullptr;
    hip_bfloat16* d_k3_tail_out = nullptr;
    hip_bfloat16* d_ref = nullptr;
    int32_t* d_row_expert = nullptr;
    int32_t* d_problem_size = nullptr;
    int32_t* d_symm_route_scratch = nullptr;
    int32_t* d_symm_grid_barrier = nullptr;
    int64_t* d_row_output_ptrs = nullptr;
    uint8_t* d_k3_local_topk_mask = nullptr;
    int32_t* d_k3_tail_tokens = nullptr;

    int target_device = 0;
    if (use_symm_buffers) {
        int device_count = 0;
        HIP_CHECK(hipGetDeviceCount(&device_count));
        if (device_count <= 0) {
            std::cerr << "No HIP devices visible for V2 symm communication" << std::endl;
            return 2;
        }
        if (opt.symm_devices > device_count) {
            std::cerr << "Requested --symm-devices " << opt.symm_devices
                      << " but only " << device_count << " HIP devices are visible" << std::endl;
            return 2;
        }
        target_device = opt.rank_idx % opt.symm_devices;
        HIP_CHECK(hipSetDevice(target_device));
        if (opt.symm_devices > 1) {
            for (int peer = 0; peer < opt.symm_devices; ++peer) {
                if (peer == target_device)
                    continue;
                int can_access = 0;
                HIP_CHECK(hipDeviceCanAccessPeer(&can_access, target_device, peer));
                if (!can_access) {
                    std::cerr << "Device " << target_device
                              << " cannot peer-read device " << peer << std::endl;
                    return 2;
                }
                hipError_t peer_status = hipDeviceEnablePeerAccess(peer, 0);
                if (peer_status != hipSuccess &&
                    peer_status != hipErrorPeerAccessAlreadyEnabled) {
                    std::cerr << "hipDeviceEnablePeerAccess(" << peer << ") failed: "
                              << hipGetErrorString(peer_status) << std::endl;
                    return 2;
                }
            }
        }
    }

    HIP_CHECK(hipMalloc(&d_x, h_x.size()));
    if (use_symm_buffers) {
        if (opt.symm_devices == 1) {
            HIP_CHECK(hipMalloc(
                &d_symm_buffers,
                static_cast<size_t>(h_symm_buffers.size())));
            d_local_symm_buffer =
                d_symm_buffers + static_cast<int64_t>(opt.rank_idx) * symm_buffer_bytes;
        } else {
            d_symm_rank_buffers.assign(opt.symm_ranks, nullptr);
            for (int rank = 0; rank < opt.symm_ranks; ++rank) {
                HIP_CHECK(hipSetDevice(rank % opt.symm_devices));
                HIP_CHECK(hipMalloc(
                    &d_symm_rank_buffers[rank],
                    static_cast<size_t>(symm_buffer_bytes)));
            }
            HIP_CHECK(hipSetDevice(target_device));
            d_local_symm_buffer = d_symm_rank_buffers[opt.rank_idx];
        }
        if (use_lowlat_symm_stage_c || use_c_symm_row_stage ||
            use_k3_copy_stage || use_k3_tail_reduce) {
            const int64_t symm_route_scratch_ints =
                32 + 2 * static_cast<int64_t>(opt.experts) * rows_aligned_per_expert;
            if (use_lowlat_symm_stage_c) {
                HIP_CHECK(hipMalloc(
                    &d_symm_route_scratch,
                    symm_route_scratch_ints *
                        static_cast<int64_t>(sizeof(int32_t))));
            }
            const int64_t barrier_ints =
                use_k3_copy_stage
                    ? static_cast<int64_t>((opt.n + 255) / 256) *
                              static_cast<int64_t>(
                                  (opt.m + opt.c_tile_n - 1) /
                                  opt.c_tile_n) +
                          2
                    : use_c_symm_row_stage
                    ? 2 * static_cast<int64_t>(
                              (opt.m + opt.c_tile_n - 1) / opt.c_tile_n)
                    : 2;
            HIP_CHECK(hipMalloc(
                &d_symm_grid_barrier,
                barrier_ints * static_cast<int64_t>(sizeof(int32_t))));
            HIP_CHECK(hipMemset(
                d_symm_grid_barrier, 0,
                barrier_ints * static_cast<int64_t>(sizeof(int32_t))));
        }
        if (use_lowlat_symm_stage_c || use_c_symm_stage) {
            HIP_CHECK(hipMalloc(&d_symm_staged_x, h_x.size()));
            HIP_CHECK(hipMalloc(
                &d_symm_staged_x_scale,
                h_x_scale.size() * sizeof(float)));
        }
    }
    HIP_CHECK(hipMalloc(&d_w_pack5, h_w_pack5.size()));
    HIP_CHECK(hipMalloc(&d_x_scale, h_x_scale.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w_scale, h_w_scale.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, h_zero.size() * sizeof(hip_bfloat16)));
    if (use_k3_rowptr) {
        HIP_CHECK(hipMalloc(
            &d_rowptr_out, h_zero.size() * sizeof(hip_bfloat16)));
        HIP_CHECK(hipMalloc(
            &d_row_output_ptrs, h_row_expert.size() * sizeof(int64_t)));
    }
    if (use_k3_tail_reduce) {
        HIP_CHECK(hipMalloc(
            &d_k3_local_topk_mask,
            h_k3_local_topk_mask.size() * sizeof(uint8_t)));
    }
    const int64_t k3_tail_out_elems =
        static_cast<int64_t>(opt.tokens) * opt.n;
    const size_t k3_tail_out_bytes =
        static_cast<size_t>(k3_tail_out_elems) * sizeof(hip_bfloat16);
    if (use_k3_tail_out) {
        HIP_CHECK(hipMalloc(&d_k3_tail_out, k3_tail_out_bytes));
        HIP_CHECK(hipMalloc(
            &d_k3_tail_tokens,
            std::max<size_t>(1, h_k3_tail_tokens.size()) *
                sizeof(int32_t)));
    }
    HIP_CHECK(hipMalloc(&d_ref, h_zero.size() * sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_row_expert, h_row_expert.size() * sizeof(int32_t)));
    if (use_lowlat_c || use_lowlat_symm_stage_c)
        HIP_CHECK(hipMalloc(&d_problem_size, h_problem_size.size() * sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size(), hipMemcpyHostToDevice));
    if (use_symm_buffers) {
        std::vector<int64_t> h_peer_ptrs(opt.symm_ranks);
        for (int rank = 0; rank < opt.symm_ranks; ++rank) {
            h_peer_ptrs[rank] = reinterpret_cast<int64_t>(
                opt.symm_devices == 1
                    ? d_symm_buffers + static_cast<int64_t>(rank) * symm_buffer_bytes
                    : d_symm_rank_buffers[rank]);
        }
        for (int rank = 0; rank < opt.symm_ranks; ++rank) {
            uint8_t* base =
                h_symm_buffers.data() + static_cast<int64_t>(rank) * symm_buffer_bytes;
            std::memcpy(
                base + deep_gemm::mega::dcu_sym_buffer_ptrs_offset(),
                h_peer_ptrs.data(),
                static_cast<int64_t>(opt.symm_ranks) * sizeof(int64_t));
        }
        if (opt.symm_devices == 1) {
            HIP_CHECK(hipMemcpy(
                d_symm_buffers, h_symm_buffers.data(), h_symm_buffers.size(),
                hipMemcpyHostToDevice));
        } else {
            for (int rank = 0; rank < opt.symm_ranks; ++rank) {
                HIP_CHECK(hipSetDevice(rank % opt.symm_devices));
                HIP_CHECK(hipMemcpy(
                    d_symm_rank_buffers[rank],
                    h_symm_buffers.data() +
                        static_cast<int64_t>(rank) * symm_buffer_bytes,
                    static_cast<size_t>(symm_buffer_bytes),
                    hipMemcpyHostToDevice));
            }
            HIP_CHECK(hipSetDevice(target_device));
        }
    }
    HIP_CHECK(hipMemcpy(
        d_w_pack5, h_w_pack5.data(), h_w_pack5.size(),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x_scale, h_x_scale.data(), h_x_scale.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w_scale, h_w_scale.data(), h_w_scale.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out, h_zero.data(), h_zero.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    if (d_k3_tail_out != nullptr) {
        HIP_CHECK(hipMemset(d_k3_tail_out, 0, k3_tail_out_bytes));
    }
    if (use_k3_rowptr) {
        HIP_CHECK(hipMemcpy(
            d_rowptr_out, h_zero.data(),
            h_zero.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
        std::vector<int64_t> h_row_output_ptrs(h_row_expert.size(), 0);
        if (use_k3_symm_combine) {
            const int num_global_experts = opt.experts * opt.symm_ranks;
            const int64_t combine_offset =
                deep_gemm::mega::combine_token_offset(
                    opt.symm_ranks, num_global_experts, opt.tokens, opt.topk,
                    symm_layout_hidden);
            for (int row = 0; row < opt.m; ++row) {
                if (h_row_expert[row] < 0)
                    continue;
                const int source_rank = h_k3_row_source_rank[row];
                const int source_token = h_k3_row_source_token[row];
                const int topk_slot = h_k3_row_topk_slot[row];
                const int combine_row = h_k3_row_combine_row[row];
                if (source_rank < 0 || source_token < 0 || topk_slot < 0 ||
                    combine_row < 0)
                    continue;
                uint8_t* rank_base =
                    opt.symm_devices == 1
                        ? d_symm_buffers +
                              static_cast<int64_t>(source_rank) *
                                  symm_buffer_bytes
                        : d_symm_rank_buffers[source_rank];
                auto* combine_base = reinterpret_cast<uint16_t*>(
                    rank_base + combine_offset);
                h_row_output_ptrs[row] = reinterpret_cast<int64_t>(
                    combine_base + static_cast<int64_t>(combine_row) * opt.n);
            }
        } else {
            for (int row = 0; row < opt.m; ++row) {
                if (h_row_expert[row] < 0)
                    continue;
                h_row_output_ptrs[row] = reinterpret_cast<int64_t>(
                    d_rowptr_out + static_cast<int64_t>(row) * opt.n);
            }
        }
        HIP_CHECK(hipMemcpy(
            d_row_output_ptrs, h_row_output_ptrs.data(),
            h_row_output_ptrs.size() * sizeof(int64_t),
            hipMemcpyHostToDevice));
    }
    if (use_k3_tail_reduce) {
        HIP_CHECK(hipMemcpy(
            d_k3_local_topk_mask, h_k3_local_topk_mask.data(),
            h_k3_local_topk_mask.size() * sizeof(uint8_t),
            hipMemcpyHostToDevice));
    }
    if (d_k3_tail_tokens != nullptr && !h_k3_tail_tokens.empty()) {
        HIP_CHECK(hipMemcpy(
            d_k3_tail_tokens, h_k3_tail_tokens.data(),
            h_k3_tail_tokens.size() * sizeof(int32_t),
            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemcpy(d_ref, h_zero.data(), h_zero.size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_expert, h_row_expert.data(), h_row_expert.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    if (use_lowlat_c || use_lowlat_symm_stage_c)
        HIP_CHECK(hipMemcpy(
            d_problem_size, h_problem_size.data(),
            h_problem_size.size() * sizeof(int32_t),
            hipMemcpyHostToDevice));

    const int c_compute_waves = opt.c_tile_n / 32;
    const int c_loader_waves = opt.c_tile_n == 64 ? 2 : 4;
    const int c_local_work_size = (c_compute_waves + c_loader_waves) * 64;
    const int c_grid_x = (opt.n + 255) / 256;
    const int c_grid_y = (opt.m + opt.c_tile_n - 1) / opt.c_tile_n;
    const int c_stage_n_group =
        use_c_symm_stage ? opt.c_stage_n_group : 1;
    const int c_launch_grid_x =
        use_c_symm_stage ? (c_grid_x + c_stage_n_group - 1) / c_stage_n_group
                         : c_grid_x;
    if (opt.print_shape || opt.touch_check) {
        std::cout << "derived_shape"
                  << " valid_rows=" << valid_rows
                  << " rows_aligned_per_expert=" << rows_aligned_per_expert
                  << " c_grid_x=" << c_grid_x
                  << " c_launch_grid_x=" << c_launch_grid_x
                  << " c_grid_y=" << c_grid_y
                  << " c_wg_count=" << c_grid_x * c_grid_y
                  << " c_launch_wg_count=" << c_launch_grid_x * c_grid_y
                  << " c_tile_n=" << opt.c_tile_n
                  << " c_local_work_size=" << c_local_work_size
                  << std::endl;
        if (use_c_symm_stage) {
            int occ_row_stage4 = 0;
            HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &occ_row_stage4,
                V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                    256, 256, true, true, 4>,
                c_local_work_size, 0));
            std::cout << "occupancy_probe"
                      << " c_symm_row_stage4_blocks_per_cu="
                      << occ_row_stage4
                      << std::endl;
        }
    }

    int c_launch_epoch = 0;
    auto launch_c = [&]() {
        if (opt.c_tile_n != 256) {
            std::cerr << "V2 large C path requires --c-tile-n 256" << std::endl;
            std::exit(2);
        }
        const int c_epoch = ++c_launch_epoch;
        const dim3 block(c_local_work_size);
        const int c_k3_copy_rows =
            use_k3_copy_stage
                ? (opt.k3_copy_workers + c_launch_grid_x - 1) /
                      c_launch_grid_x
                : 0;
        const int c_kernel_grid_y =
            c_grid_y + c_k3_copy_rows;
        const dim3 grid(c_launch_grid_x, c_kernel_grid_y);
        if (use_c_symm_stage) {
            V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                256, 256, true, true, 4>
                <<<grid, block>>>(
                d_out, d_symm_staged_x, d_w_pack5,
                d_symm_staged_x_scale, d_w_scale, d_row_expert, opt.m,
                opt.n, opt.k, rows_aligned_per_expert,
                opt.valid_rows_per_expert, d_local_symm_buffer,
                d_symm_grid_barrier, c_epoch, opt.rank_idx, opt.symm_ranks,
                opt.experts * opt.symm_ranks, opt.tokens, opt.topk,
                opt.tokens, nullptr);
        } else if (opt.k == 2048) {
            if (use_k3_copy_stage) {
                V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                    256, 256, true, false, 1, 4096, 2048, false, true>
                    <<<grid, block>>>(
                    d_out, d_x, d_w_pack5, d_x_scale, d_w_scale,
                    d_row_expert, opt.m, opt.n, opt.k,
                    rows_aligned_per_expert, opt.valid_rows_per_expert,
                    d_local_symm_buffer, d_symm_grid_barrier, c_epoch,
                    opt.rank_idx, opt.symm_ranks,
                    opt.experts * opt.symm_ranks, opt.tokens, opt.topk,
                    -1, d_row_output_ptrs, opt.k3_copy_workers,
                    opt.k3_tail_reduce, d_k3_local_topk_mask,
                    d_k3_tail_out, d_k3_tail_tokens,
                    static_cast<int>(h_k3_tail_tokens.size()));
            } else if (use_k3_rowptr) {
                V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                    256, 256, true, false, 1, 4096, 2048, true>
                    <<<grid, block>>>(
                    d_out, d_x, d_w_pack5, d_x_scale, d_w_scale,
                    d_row_expert, opt.m, opt.n, opt.k,
                    rows_aligned_per_expert, opt.valid_rows_per_expert,
                    nullptr, nullptr, 0, 0, 1, opt.experts, opt.tokens,
                    opt.topk, -1, d_row_output_ptrs);
            } else {
                V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                    256, 256, true, false, 1, 4096, 2048>
                    <<<grid, block>>>(
                    d_out, d_x, d_w_pack5, d_x_scale, d_w_scale,
                    d_row_expert, opt.m, opt.n, opt.k,
                    rows_aligned_per_expert, opt.valid_rows_per_expert,
                    nullptr, nullptr, 0, 0, 1, opt.experts, opt.tokens,
                    opt.topk, -1, nullptr);
            }
        } else {
            V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                256, 256, true>
                <<<grid, block>>>(
                d_out, d_x, d_w_pack5, d_x_scale, d_w_scale, d_row_expert,
                opt.m, opt.n, opt.k, rows_aligned_per_expert,
                opt.valid_rows_per_expert, nullptr, nullptr, 0, 0, 1,
                opt.experts, opt.tokens, opt.topk, -1, nullptr);
        }
    };

    auto launch_lowlat_c = [&]() {
        const dim3 block(256);
#define K1_LAUNCH_LOWLAT_IMPL(PROBLEM_K, BLOCK_M, CUS, MASK_TINY_STORE, USE_STAGE, USE_ROWPTR) \
        do {                                                                    \
            V2_K1_LowLatencyMaskedGroupGemmKernel<                                 \
                32, 4096, PROBLEM_K, BLOCK_M, 256, 64, 4, CUS,                 \
                MASK_TINY_STORE, USE_STAGE, USE_ROWPTR>                         \
                <<<dim3(CUS), block>>>(                                         \
                d_out,                                                          \
                USE_STAGE ? d_symm_staged_x : d_x,                              \
                d_w_pack5,                                                   \
                USE_STAGE ? d_symm_staged_x_scale : d_x_scale,                  \
                d_w_scale,                                                       \
                d_problem_size, rows_aligned_per_expert,                         \
                (USE_STAGE || (USE_ROWPTR && use_k3_tail_reduce)) ? d_local_symm_buffer : nullptr, \
                USE_STAGE ? d_symm_route_scratch : nullptr,                    \
                (USE_STAGE || (USE_ROWPTR && use_k3_tail_reduce)) ? d_symm_grid_barrier : nullptr, \
                opt.rank_idx, opt.symm_ranks, opt.experts * opt.symm_ranks,      \
                opt.tokens, opt.topk, USE_STAGE ? opt.tokens : -1,               \
                USE_ROWPTR ? d_row_output_ptrs : nullptr,                        \
                (USE_ROWPTR && use_k3_tail_reduce) ? opt.k3_tail_reduce : 0);    \
        } while (0)
#define K1_LAUNCH_LOWLAT(BLOCK_M, CUS, MASK_TINY_STORE)                         \
        do {                                                                    \
            const int problem_k = opt.k;                                        \
            if (use_lowlat_symm_stage_c) {                                       \
                K1_LAUNCH_LOWLAT_IMPL(4096, BLOCK_M, CUS, MASK_TINY_STORE, true, false); \
            } else {                                                            \
                if (problem_k == 2048) {                                         \
                    if (use_k3_rowptr) {                                        \
                        K1_LAUNCH_LOWLAT_IMPL(2048, BLOCK_M, CUS, MASK_TINY_STORE, false, true); \
                    } else {                                                    \
                        K1_LAUNCH_LOWLAT_IMPL(2048, BLOCK_M, CUS, MASK_TINY_STORE, false, false); \
                    }                                                           \
                } else {                                                        \
                    K1_LAUNCH_LOWLAT_IMPL(4096, BLOCK_M, CUS, MASK_TINY_STORE, false, false); \
                }                                                               \
            }                                                                   \
        } while (0)
        if (opt.ll_block_m == 48 && opt.ll_cus == 64) {
            K1_LAUNCH_LOWLAT(48, 64, false);
        } else if (opt.ll_block_m == 48 && opt.ll_cus == 128) {
            K1_LAUNCH_LOWLAT(48, 128, false);
        } else if (opt.ll_block_m == 64 && opt.ll_cus == 64) {
            K1_LAUNCH_LOWLAT(64, 64, false);
        } else if (opt.ll_block_m == 64 && opt.ll_cus == 128) {
            K1_LAUNCH_LOWLAT(64, 128, false);
        } else if (opt.ll_block_m == 32 && opt.ll_cus == 64) {
            if (opt.valid_rows_per_expert <= 16) {
                K1_LAUNCH_LOWLAT(32, 64, true);
            } else {
                K1_LAUNCH_LOWLAT(32, 64, false);
            }
        } else if (opt.ll_block_m == 32 && opt.ll_cus == 128) {
            K1_LAUNCH_LOWLAT(32, 128, false);
        } else if (opt.ll_block_m == 16 && opt.ll_cus == 32) {
            K1_LAUNCH_LOWLAT(16, 32, false);
        } else if (opt.ll_block_m == 16 && opt.ll_cus == 64) {
            K1_LAUNCH_LOWLAT(16, 64, false);
        } else if (opt.ll_block_m == 16 && opt.ll_cus == 128) {
            K1_LAUNCH_LOWLAT(16, 128, false);
        } else {
            std::cerr << "Expected --ll-block-m/--ll-cus to be 16/{32,64,128}, 32/{64,128}, 48/{64,128}, or 64/{64,128}" << std::endl;
            std::exit(2);
        }
#undef K1_LAUNCH_LOWLAT
#undef K1_LAUNCH_LOWLAT_IMPL
    };

    auto launch_selected = [&]() {
        if (opt.mode == Mode::kLowLatencyC ||
            opt.mode == Mode::kLowLatencySymmStageC)
            launch_lowlat_c();
        else
            launch_c();
    };

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    const int measure_rounds = std::max(1, opt.measure_rounds);
    double avg_ms = 0.0;

    auto time_repeats = [&](auto&& launch_fn) -> double {
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < opt.repeat; ++i)
            launch_fn();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        return static_cast<double>(elapsed_ms) / opt.repeat;
    };

    auto print_one = [&](const char* label, double ms) {
        const double padded =
            2.0 * static_cast<double>(opt.m) * opt.n * opt.k / (ms * 1.0e9);
        const double valid = valid_tflops(valid_rows, opt.n, opt.k, ms);
        std::cout << label
                  << "_avg_ms=" << ms
                  << " padded_tflops=" << padded
                  << " valid_tflops=" << valid
                  << " c_wg_count=" << c_grid_x * c_grid_y
                  << " c_launch_wg_count=" << c_launch_grid_x * c_grid_y;
    };

    for (int i = 0; i < opt.warmup; ++i)
        launch_selected();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    for (int round = 0; round < measure_rounds; ++round) {
        avg_ms = time_repeats(launch_selected);
        if (measure_rounds > 1)
            std::cout << "measure_round=" << (round + 1) << " ";
        const char* label =
            opt.mode == Mode::kLowLatencyC
                ? "c_ll"
                : (opt.mode == Mode::kLowLatencySymmStageC
                       ? "c_ll_symm_stage"
                       : (opt.mode == Mode::kCSymmStage
                              ? "c_symm_stage"
                              : "c"));
        print_one(label, avg_ms);
        std::cout << std::endl;
    }

    auto copy_k3_combine_to_host_rows =
        [&](std::vector<hip_bfloat16>& dst) {
            std::memset(
                dst.data(), 0, dst.size() * sizeof(hip_bfloat16));
            if (!use_k3_symm_combine)
                return;
            const int num_global_experts = opt.experts * opt.symm_ranks;
            const int64_t combine_offset =
                deep_gemm::mega::combine_token_offset(
                    opt.symm_ranks, num_global_experts, opt.tokens, opt.topk,
                    symm_layout_hidden);
            std::vector<uint8_t> h_symm_after(h_symm_buffers.size(), 0);
            if (opt.symm_devices == 1) {
                HIP_CHECK(hipMemcpy(
                    h_symm_after.data(), d_symm_buffers,
                    h_symm_after.size(), hipMemcpyDeviceToHost));
            } else {
                for (int rank = 0; rank < opt.symm_ranks; ++rank) {
                    HIP_CHECK(hipSetDevice(rank % opt.symm_devices));
                    HIP_CHECK(hipMemcpy(
                        h_symm_after.data() +
                            static_cast<int64_t>(rank) * symm_buffer_bytes,
                        d_symm_rank_buffers[rank],
                        static_cast<size_t>(symm_buffer_bytes),
                        hipMemcpyDeviceToHost));
                }
                HIP_CHECK(hipSetDevice(target_device));
            }
            for (int row = 0; row < opt.m; ++row) {
                if (h_row_expert[row] < 0)
                    continue;
                const int source_rank = h_k3_row_source_rank[row];
                const int source_token = h_k3_row_source_token[row];
                const int topk_slot = h_k3_row_topk_slot[row];
                const int combine_row = h_k3_row_combine_row[row];
                if (source_rank < 0 || source_token < 0 || topk_slot < 0 ||
                    combine_row < 0)
                    continue;
                const auto* combine = reinterpret_cast<const uint16_t*>(
                    h_symm_after.data() +
                    static_cast<int64_t>(source_rank) * symm_buffer_bytes +
                    combine_offset);
                for (int col = 0; col < opt.n; ++col) {
                    const uint16_t bits =
                        combine[static_cast<int64_t>(combine_row) * opt.n +
                                col];
                    std::memcpy(
                        &dst[static_cast<int64_t>(row) * opt.n + col],
                        &bits, sizeof(bits));
                }
            }
        };

    auto copy_k3_tail_reduce_to_host =
        [&](std::vector<hip_bfloat16>& dst) {
            std::memset(dst.data(), 0, dst.size() * sizeof(hip_bfloat16));
            if (d_k3_tail_out != nullptr) {
                HIP_CHECK(hipMemcpy(
                    dst.data(), d_k3_tail_out, k3_tail_out_bytes,
                    hipMemcpyDeviceToHost));
            } else {
                HIP_CHECK(hipMemcpy(
                    dst.data(), d_out,
                    dst.size() * sizeof(hip_bfloat16),
                    hipMemcpyDeviceToHost));
            }
        };

    if (opt.touch_check) {
        std::vector<hip_bfloat16> h_touch(h_zero.size());
        if (use_k3_symm_combine) {
            copy_k3_combine_to_host_rows(h_touch);
        } else {
            HIP_CHECK(hipMemcpy(
                h_touch.data(), use_k3_rowptr ? d_rowptr_out : d_out,
                h_touch.size() * sizeof(hip_bfloat16),
                hipMemcpyDeviceToHost));
        }
        int64_t touched_rows = 0;
        int64_t touched_valid_rows = 0;
        int64_t touched_padding_rows = 0;
        int64_t touched_row_tiles = 0;
        for (int row = 0; row < opt.m; ++row) {
            bool row_touched = false;
            const int64_t row_base = static_cast<int64_t>(row) * opt.n;
            for (int col = 0; col < opt.n; ++col) {
                if (bf16_bits(h_touch[row_base + col]) != 0) {
                    row_touched = true;
                    break;
                }
            }
            if (!row_touched)
                continue;
            ++touched_rows;
            if (h_row_expert[row] >= 0) {
                ++touched_valid_rows;
            } else {
                ++touched_padding_rows;
            }
        }
        for (int row_tile = 0; row_tile < c_grid_y; ++row_tile) {
            bool tile_touched = false;
            const int row_begin = row_tile * opt.c_tile_n;
            const int row_end = std::min(row_begin + opt.c_tile_n, opt.m);
            for (int row = row_begin; row < row_end && !tile_touched; ++row) {
                const int64_t row_base = static_cast<int64_t>(row) * opt.n;
                for (int col = 0; col < opt.n; ++col) {
                    if (bf16_bits(h_touch[row_base + col]) != 0) {
                        tile_touched = true;
                        break;
                    }
                }
            }
            touched_row_tiles += tile_touched ? 1 : 0;
        }
        std::cout << "touch_check"
                  << " touched_rows=" << touched_rows
                  << " touched_valid_rows=" << touched_valid_rows
                  << " touched_padding_rows=" << touched_padding_rows
                  << " touched_row_tiles=" << touched_row_tiles
                  << " expected_rows=" << opt.m
                  << " expected_valid_rows=" << valid_rows
                  << " expected_row_tiles=" << c_grid_y
                  << std::endl;
    }

    if (opt.check && opt.c_tile_n != 256) {
        const int ref_rows_aligned_per_expert =
            ((std::max(1, opt.valid_rows_per_expert) + 255) / 256) * 256;
        const int ref_m = ref_rows_aligned_per_expert * opt.experts;
        std::vector<uint8_t> h_x_ref(static_cast<int64_t>(ref_m) * opt.k, 0);
        std::vector<float> h_x_scale_ref(ref_m, 0.0f);
        std::vector<hip_bfloat16> h_zero_ref(static_cast<int64_t>(ref_m) * opt.n);
        const auto h_row_expert_ref =
            make_row_expert(ref_m, opt.experts, opt.valid_rows_per_expert);

        for (int e = 0; e < opt.experts; ++e) {
            for (int r = 0; r < opt.valid_rows_per_expert; ++r) {
                const int src_row = e * rows_aligned_per_expert + r;
                const int dst_row = e * ref_rows_aligned_per_expert + r;
                std::memcpy(
                    h_x_ref.data() + static_cast<int64_t>(dst_row) * opt.k,
                    h_x.data() + static_cast<int64_t>(src_row) * opt.k,
                    opt.k);
                h_x_scale_ref[dst_row] = h_x_scale[src_row];
            }
        }

        uint8_t* d_x_ref_compact = nullptr;
        float* d_x_scale_ref_compact = nullptr;
        hip_bfloat16* d_ref_compact = nullptr;
        int32_t* d_row_expert_ref_compact = nullptr;
        HIP_CHECK(hipMalloc(&d_x_ref_compact, h_x_ref.size()));
        HIP_CHECK(hipMalloc(
            &d_x_scale_ref_compact,
            h_x_scale_ref.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(
            &d_ref_compact,
            h_zero_ref.size() * sizeof(hip_bfloat16)));
        HIP_CHECK(hipMalloc(
            &d_row_expert_ref_compact,
            h_row_expert_ref.size() * sizeof(int32_t)));
        HIP_CHECK(hipMemcpy(
            d_x_ref_compact, h_x_ref.data(), h_x_ref.size(),
            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(
            d_x_scale_ref_compact, h_x_scale_ref.data(),
            h_x_scale_ref.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(
            d_ref_compact, h_zero_ref.data(),
            h_zero_ref.size() * sizeof(hip_bfloat16),
            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(
            d_row_expert_ref_compact, h_row_expert_ref.data(),
            h_row_expert_ref.size() * sizeof(int32_t),
            hipMemcpyHostToDevice));

        const dim3 ref_block(768);
        const dim3 ref_grid((opt.n + 255) / 256, (ref_m + 255) / 256);
        if (opt.k == 2048) {
            V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                256, 256, true, false, 1, 4096, 2048>
                <<<ref_grid, ref_block>>>(
                d_ref_compact, d_x_ref_compact, d_w_pack5,
                d_x_scale_ref_compact, d_w_scale, d_row_expert_ref_compact,
                ref_m, opt.n, opt.k, ref_rows_aligned_per_expert,
                opt.valid_rows_per_expert, nullptr, nullptr, 0, 0, 1,
                opt.experts, opt.tokens, opt.topk, -1, nullptr);
        } else {
            V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                256, 256, true>
                <<<ref_grid, ref_block>>>(
                d_ref_compact, d_x_ref_compact, d_w_pack5,
                d_x_scale_ref_compact, d_w_scale, d_row_expert_ref_compact,
                ref_m, opt.n, opt.k, ref_rows_aligned_per_expert,
                opt.valid_rows_per_expert, nullptr, nullptr, 0, 0, 1,
                opt.experts, opt.tokens, opt.topk, -1, nullptr);
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<hip_bfloat16> h_out(h_zero.size());
        std::vector<hip_bfloat16> h_ref_compact(h_zero_ref.size());
        if (use_k3_tail_reduce) {
            copy_k3_tail_reduce_to_host(h_out);
        } else if (use_k3_symm_combine) {
            copy_k3_combine_to_host_rows(h_out);
        } else {
            HIP_CHECK(hipMemcpy(
                h_out.data(), use_k3_rowptr ? d_rowptr_out : d_out,
                h_out.size() * sizeof(hip_bfloat16),
                hipMemcpyDeviceToHost));
        }
        HIP_CHECK(hipMemcpy(
            h_ref_compact.data(), d_ref_compact,
            h_ref_compact.size() * sizeof(hip_bfloat16),
            hipMemcpyDeviceToHost));

        if (opt.debug_values) {
            std::cout << "debug_row0_first64";
            for (int col = 0; col < std::min(opt.n, 64); ++col) {
                const float got = bf16_to_float(h_out[col]);
                const float expect = bf16_to_float(h_ref_compact[col]);
                std::cout << " [" << col << ":" << got << "/" << expect << "]";
            }
            std::cout << std::endl;
        }

        if (use_k3_tail_reduce) {
            std::vector<float> h_tail_ref(
                static_cast<int64_t>(opt.tokens) * opt.n, 0.0f);
            for (int e = 0; e < opt.experts; ++e) {
                for (int r = 0; r < opt.valid_rows_per_expert; ++r) {
                    const int row = e * rows_aligned_per_expert + r;
                    if (h_k3_row_source_rank[row] != opt.rank_idx)
                        continue;
                    const int source_token = h_k3_row_source_token[row];
                    if (source_token < 0 || source_token >= opt.tokens)
                        continue;
                    const int ref_row = e * ref_rows_aligned_per_expert + r;
                    for (int col = 0; col < opt.n; ++col) {
                        h_tail_ref[static_cast<int64_t>(source_token) * opt.n +
                                   col] +=
                            bf16_to_float(
                                h_ref_compact[static_cast<int64_t>(ref_row) *
                                                  opt.n +
                                              col]);
                    }
                }
            }

            double max_abs = 0.0;
            double mean_abs = 0.0;
            int64_t count = 0;
            int64_t bit_mismatch = 0;
            int64_t value_mismatch = 0;
            int printed = 0;
            for (int token = 0; token < opt.tokens; ++token) {
                for (int col = 0; col < opt.n; ++col) {
                    const int64_t idx =
                        static_cast<int64_t>(token) * opt.n + col;
                    const hip_bfloat16 expect_bf16 =
                        float_to_bf16_host(h_tail_ref[idx]);
                    const float got = bf16_to_float(h_out[idx]);
                    const float expect = bf16_to_float(expect_bf16);
                    const uint16_t got_bits = bf16_bits(h_out[idx]);
                    const uint16_t expect_bits = bf16_bits(expect_bf16);
                    const double diff = std::abs(
                        static_cast<double>(got) -
                        static_cast<double>(expect));
                    const bool bad_value =
                        !std::isfinite(got) || !std::isfinite(expect) ||
                        !std::isfinite(diff);
                    const bool bits_differ = got_bits != expect_bits;
                    const bool value_differs =
                        bad_value || diff > opt.allowed_max_abs;
                    bit_mismatch += bits_differ ? 1 : 0;
                    value_mismatch += value_differs ? 1 : 0;
                    if ((value_differs || bits_differ) && printed < 8) {
                        std::cout << "mismatch token=" << token
                                  << " col=" << col
                                  << " got=" << got
                                  << " ref=" << expect
                                  << " got_bits=0x" << std::hex << got_bits
                                  << " ref_bits=0x" << expect_bits << std::dec
                                  << " diff=" << diff << std::endl;
                        ++printed;
                    }
                    max_abs = bad_value
                                  ? std::numeric_limits<double>::infinity()
                                  : std::max(max_abs, diff);
                    mean_abs += diff;
                    ++count;
                }
            }
            mean_abs = count > 0 ? mean_abs / static_cast<double>(count) : 0.0;
            std::cout << "correctness_ref=c_tail_reduce_local_rank"
                      << " max_abs=" << max_abs
                      << " mean_abs=" << mean_abs
                      << " bit_mismatch=" << bit_mismatch
                      << " value_mismatch=" << value_mismatch
                      << " allowed_max_abs=" << opt.allowed_max_abs
                      << " checked=" << count << std::endl;

            HIP_CHECK(hipFree(d_x_ref_compact));
            HIP_CHECK(hipFree(d_x_scale_ref_compact));
            HIP_CHECK(hipFree(d_ref_compact));
            HIP_CHECK(hipFree(d_row_expert_ref_compact));
            if (value_mismatch != 0) {
                std::cerr << "FAIL correctness" << std::endl;
                return 1;
            }
            std::cout << "PASS correctness" << std::endl;
        } else {
        double max_abs = 0.0;
        double mean_abs = 0.0;
        int64_t count = 0;
        int64_t bit_mismatch = 0;
        int64_t value_mismatch = 0;
        int printed = 0;
        for (int e = 0; e < opt.experts; ++e) {
            for (int r = 0; r < opt.valid_rows_per_expert; ++r) {
                const int row = e * rows_aligned_per_expert + r;
                const int ref_row = e * ref_rows_aligned_per_expert + r;
                for (int col = 0; col < opt.n; ++col) {
                    const int64_t idx = static_cast<int64_t>(row) * opt.n + col;
                    const int64_t ref_idx =
                        static_cast<int64_t>(ref_row) * opt.n + col;
                    const float got = bf16_to_float(h_out[idx]);
                    const float expect = bf16_to_float(h_ref_compact[ref_idx]);
                    const uint16_t got_bits = bf16_bits(h_out[idx]);
                    const uint16_t expect_bits =
                        bf16_bits(h_ref_compact[ref_idx]);
                    const double diff = std::abs(
                        static_cast<double>(got) -
                        static_cast<double>(expect));
                    const bool bad_value =
                        !std::isfinite(got) || !std::isfinite(expect) ||
                        !std::isfinite(diff);
                    const bool bits_differ = got_bits != expect_bits;
                    const bool value_differs =
                        bad_value || diff > opt.allowed_max_abs;
                    bit_mismatch += bits_differ ? 1 : 0;
                    value_mismatch += value_differs ? 1 : 0;
                    if ((value_differs || bits_differ) && printed < 8) {
                        std::cout << "mismatch row=" << row
                                  << " ref_row=" << ref_row
                                  << " col=" << col
                                  << " expert=" << e
                                  << " got=" << got
                                  << " ref=" << expect
                                  << " got_bits=0x" << std::hex << got_bits
                                  << " ref_bits=0x" << expect_bits << std::dec
                                  << " diff=" << diff << std::endl;
                        ++printed;
                    }
                    max_abs = bad_value
                                  ? std::numeric_limits<double>::infinity()
                                  : std::max(max_abs, diff);
                    mean_abs += diff;
                    ++count;
                }
            }
        }
        mean_abs = count > 0 ? mean_abs / static_cast<double>(count) : 0.0;
        std::cout << "correctness_ref=c_mt256_remap"
                  << " ref_rows_aligned_per_expert="
                  << ref_rows_aligned_per_expert
                  << " max_abs=" << max_abs
                  << " mean_abs=" << mean_abs
                  << " bit_mismatch=" << bit_mismatch
                  << " value_mismatch=" << value_mismatch
                  << " allowed_max_abs=" << opt.allowed_max_abs
                  << " checked=" << count << std::endl;

        HIP_CHECK(hipFree(d_x_ref_compact));
        HIP_CHECK(hipFree(d_x_scale_ref_compact));
        HIP_CHECK(hipFree(d_ref_compact));
        HIP_CHECK(hipFree(d_row_expert_ref_compact));
        if (value_mismatch != 0) {
            std::cerr << "FAIL correctness" << std::endl;
            return 1;
        }
        std::cout << "PASS correctness" << std::endl;
        }
    } else if (opt.check) {
        if (opt.debug_values && use_c_symm_stage) {
            const int debug_row =
                opt.debug_values > 1 ? opt.debug_values : 0;
            if (debug_row >= 0 && debug_row < opt.m) {
                std::vector<uint8_t> h_staged_row(opt.k);
                std::vector<float> h_staged_scale(1);
                HIP_CHECK(hipMemcpy(
                    h_staged_row.data(),
                    d_symm_staged_x + static_cast<int64_t>(debug_row) * opt.k,
                    h_staged_row.size(), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(
                    h_staged_scale.data(), d_symm_staged_x_scale + debug_row,
                    sizeof(float), hipMemcpyDeviceToHost));
                int byte_mismatch = 0;
                int first_bad = -1;
                for (int i = 0; i < opt.k; ++i) {
                    const uint8_t expect =
                        h_x[static_cast<int64_t>(debug_row) * opt.k + i];
                    if (h_staged_row[i] != expect) {
                        ++byte_mismatch;
                        if (first_bad < 0)
                            first_bad = i;
                    }
                }
                std::cout << "debug_staged_row"
                          << " row=" << debug_row
                          << " byte_mismatch=" << byte_mismatch
                          << " first_bad=" << first_bad
                          << " staged_scale=" << h_staged_scale[0]
                          << " expect_scale=" << h_x_scale[debug_row];
                if (first_bad >= 0) {
                    std::cout << " staged_byte="
                              << static_cast<int>(h_staged_row[first_bad])
                              << " expect_byte="
                              << static_cast<int>(
                                     h_x[static_cast<int64_t>(debug_row) *
                                             opt.k +
                                         first_bad]);
                }
                std::cout << std::endl;
            }
        }
        HIP_CHECK(hipMemcpy(
            d_ref, h_zero.data(),
            h_zero.size() * sizeof(hip_bfloat16),
            hipMemcpyHostToDevice));

        const char* correctness_ref_name = "c_pack5_baseline";
        const dim3 ref_block(c_local_work_size);
        const dim3 ref_grid(c_grid_x, c_grid_y);
        if (opt.k == 2048) {
            V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                256, 256, true, false, 1, 4096, 2048>
                <<<ref_grid, ref_block>>>(
                d_ref, d_x, d_w_pack5, d_x_scale, d_w_scale,
                d_row_expert, opt.m, opt.n, opt.k,
                rows_aligned_per_expert, opt.valid_rows_per_expert,
                nullptr, nullptr, 0, 0, 1, opt.experts, opt.tokens,
                opt.topk, -1, nullptr);
        } else {
            V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<
                256, 256, true>
                <<<ref_grid, ref_block>>>(
                d_ref, d_x, d_w_pack5, d_x_scale, d_w_scale,
                d_row_expert, opt.m, opt.n, opt.k,
                rows_aligned_per_expert, opt.valid_rows_per_expert,
                nullptr, nullptr, 0, 0, 1, opt.experts, opt.tokens,
                opt.topk, -1, nullptr);
        }
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<hip_bfloat16> h_out(h_zero.size());
        std::vector<hip_bfloat16> h_ref(h_zero.size());
        if (use_k3_tail_reduce) {
            copy_k3_tail_reduce_to_host(h_out);
        } else if (use_k3_symm_combine) {
            copy_k3_combine_to_host_rows(h_out);
        } else {
            HIP_CHECK(hipMemcpy(
                h_out.data(), use_k3_rowptr ? d_rowptr_out : d_out,
                h_out.size() * sizeof(hip_bfloat16),
                hipMemcpyDeviceToHost));
        }
        HIP_CHECK(hipMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));

        if (use_k3_tail_reduce) {
            std::vector<float> h_tail_ref(
                static_cast<int64_t>(opt.tokens) * opt.n, 0.0f);
            for (int row = 0; row < opt.m; ++row) {
                if (h_row_expert[row] < 0)
                    continue;
                if (h_k3_row_source_rank[row] != opt.rank_idx)
                    continue;
                const int source_token = h_k3_row_source_token[row];
                if (source_token < 0 || source_token >= opt.tokens)
                    continue;
                for (int col = 0; col < opt.n; ++col) {
                    h_tail_ref[static_cast<int64_t>(source_token) * opt.n +
                               col] +=
                        bf16_to_float(
                            h_ref[static_cast<int64_t>(row) * opt.n + col]);
                }
            }
            double max_abs = 0.0;
            double mean_abs = 0.0;
            int64_t count = 0;
            int64_t bit_mismatch = 0;
            int64_t value_mismatch = 0;
            int printed = 0;
            for (int token = 0; token < opt.tokens; ++token) {
                for (int col = 0; col < opt.n; ++col) {
                    const int64_t idx =
                        static_cast<int64_t>(token) * opt.n + col;
                    const hip_bfloat16 expect_bf16 =
                        float_to_bf16_host(h_tail_ref[idx]);
                    const float got = bf16_to_float(h_out[idx]);
                    const float expect = bf16_to_float(expect_bf16);
                    const uint16_t got_bits = bf16_bits(h_out[idx]);
                    const uint16_t expect_bits = bf16_bits(expect_bf16);
                    const double diff = std::abs(
                        static_cast<double>(got) -
                        static_cast<double>(expect));
                    const bool bad_value =
                        !std::isfinite(got) || !std::isfinite(expect) ||
                        !std::isfinite(diff);
                    const bool bits_differ = got_bits != expect_bits;
                    const bool value_differs =
                        bad_value || diff > opt.allowed_max_abs;
                    bit_mismatch += bits_differ ? 1 : 0;
                    value_mismatch += value_differs ? 1 : 0;
                    if ((value_differs || bits_differ) && printed < 8) {
                        std::cout << "mismatch token=" << token
                                  << " col=" << col
                                  << " got=" << got
                                  << " ref=" << expect
                                  << " got_bits=0x" << std::hex << got_bits
                                  << " ref_bits=0x" << expect_bits << std::dec
                                  << " diff=" << diff << std::endl;
                        ++printed;
                    }
                    max_abs = bad_value
                                  ? std::numeric_limits<double>::infinity()
                                  : std::max(max_abs, diff);
                    mean_abs += diff;
                    ++count;
                }
            }
            mean_abs = count > 0 ? mean_abs / static_cast<double>(count) : 0.0;
            std::cout << "correctness_ref=c_tail_reduce_local_rank"
                      << " max_abs=" << max_abs
                      << " mean_abs=" << mean_abs
                      << " bit_mismatch=" << bit_mismatch
                      << " value_mismatch=" << value_mismatch
                      << " allowed_max_abs=" << opt.allowed_max_abs
                      << " checked=" << count << std::endl;
            if (value_mismatch != 0) {
                std::cerr << "FAIL correctness" << std::endl;
                return 1;
            }
            std::cout << "PASS correctness" << std::endl;
        } else {
        double max_abs = 0.0;
        double mean_abs = 0.0;
        int64_t count = 0;
        int64_t bit_mismatch = 0;
        int64_t value_mismatch = 0;
        int printed = 0;
        for (int row = 0; row < opt.m; ++row) {
            if (h_row_expert[row] < 0)
                continue;
            for (int col = 0; col < opt.n; ++col) {
                const int64_t idx = static_cast<int64_t>(row) * opt.n + col;
                const float got = bf16_to_float(h_out[idx]);
                const float expect = bf16_to_float(h_ref[idx]);
                const uint16_t got_bits = bf16_bits(h_out[idx]);
                const uint16_t expect_bits = bf16_bits(h_ref[idx]);
                const double diff = std::abs(
                    static_cast<double>(got) - static_cast<double>(expect));
                const bool bad_value =
                    !std::isfinite(got) || !std::isfinite(expect) || !std::isfinite(diff);
                const bool bits_differ = got_bits != expect_bits;
                const bool value_differs =
                    bad_value || diff > opt.allowed_max_abs;
                bit_mismatch += bits_differ ? 1 : 0;
                value_mismatch += value_differs ? 1 : 0;
                if ((value_differs || bits_differ) && printed < 8) {
                    std::cout << "mismatch row=" << row
                              << " col=" << col
                              << " expert=" << h_row_expert[row]
                              << " got=" << got
                              << " ref=" << expect
                              << " got_bits=0x" << std::hex << got_bits
                              << " ref_bits=0x" << expect_bits << std::dec
                              << " diff=" << diff << std::endl;
                    ++printed;
                }
                max_abs = bad_value ? std::numeric_limits<double>::infinity()
                                    : std::max(max_abs, diff);
                mean_abs += diff;
                ++count;
            }
        }
        mean_abs = count > 0 ? mean_abs / static_cast<double>(count) : 0.0;
        std::cout << "correctness_ref=" << correctness_ref_name
                  << " max_abs=" << max_abs
                  << " mean_abs=" << mean_abs
                  << " bit_mismatch=" << bit_mismatch
                  << " value_mismatch=" << value_mismatch
                  << " allowed_max_abs=" << opt.allowed_max_abs
                  << " checked=" << count << std::endl;
        if (value_mismatch != 0) {
            std::cerr << "FAIL correctness" << std::endl;
            return 1;
        }
        std::cout << "PASS correctness" << std::endl;
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_x));
    if (d_symm_buffers != nullptr)
        HIP_CHECK(hipFree(d_symm_buffers));
    for (int rank = 0; rank < static_cast<int>(d_symm_rank_buffers.size()); ++rank) {
        if (d_symm_rank_buffers[rank] != nullptr) {
            HIP_CHECK(hipSetDevice(rank % opt.symm_devices));
            HIP_CHECK(hipFree(d_symm_rank_buffers[rank]));
        }
    }
    if (use_symm_buffers)
        HIP_CHECK(hipSetDevice(target_device));
    if (d_w_pack5 != nullptr)
        HIP_CHECK(hipFree(d_w_pack5));
    HIP_CHECK(hipFree(d_x_scale));
    if (d_symm_staged_x != nullptr)
        HIP_CHECK(hipFree(d_symm_staged_x));
    if (d_symm_staged_x_scale != nullptr)
        HIP_CHECK(hipFree(d_symm_staged_x_scale));
    HIP_CHECK(hipFree(d_w_scale));
    HIP_CHECK(hipFree(d_out));
    if (d_rowptr_out != nullptr)
        HIP_CHECK(hipFree(d_rowptr_out));
    if (d_k3_tail_out != nullptr)
        HIP_CHECK(hipFree(d_k3_tail_out));
    if (d_k3_tail_tokens != nullptr)
        HIP_CHECK(hipFree(d_k3_tail_tokens));
    HIP_CHECK(hipFree(d_ref));
    HIP_CHECK(hipFree(d_row_expert));
    if (d_problem_size != nullptr)
        HIP_CHECK(hipFree(d_problem_size));
    if (d_symm_route_scratch != nullptr)
        HIP_CHECK(hipFree(d_symm_route_scratch));
    if (d_symm_grid_barrier != nullptr)
        HIP_CHECK(hipFree(d_symm_grid_barrier));
    if (d_row_output_ptrs != nullptr)
        HIP_CHECK(hipFree(d_row_output_ptrs));
    if (d_k3_local_topk_mask != nullptr)
        HIP_CHECK(hipFree(d_k3_local_topk_mask));
    return 0;
}
#endif  // DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN
