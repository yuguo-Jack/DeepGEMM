#pragma once

// Stage-owned V3 K1 LL pack5 group GEMM core derived from
// hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp. V3 normal uses the separate
// ASM-pack5 dispatch path.

#include <hip/hip_bfloat16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include <deep_gemm/comm/mega_moe_dcu.cuh>
#include <deep_gemm/layout/mega_moe_dcu.cuh>
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
    int wg_n_tile) {
    return wg_n_tile < 8 ? (stage_iter ^ 16) : stage_iter;
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
    uint32_t global_offset0) {
    using LdsIntPtr = __attribute__((address_space(3))) int*;
    const auto lds_ptr = (LdsIntPtr)(lds_base + lds_first_byte_offset);
    const uint32_t lds_addr =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds_ptr));
    const uint32_t global_offset1 = global_offset0 + 1u * 0x20000u;
    const uint32_t global_offset2 = global_offset0 + 2u * 0x20000u;
    const uint32_t global_offset3 = global_offset0 + 3u * 0x20000u;
    const uint32_t global_offset4 = global_offset0 + 4u * 0x20000u;
    const uint32_t global_offset5 = global_offset0 + 5u * 0x20000u;
    const uint32_t global_offset6 = global_offset0 + 6u * 0x20000u;
    const uint32_t global_offset7 = global_offset0 + 7u * 0x20000u;
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
    uint32_t global_offset0) {
    const uint32_t global_offset1 = global_offset0 + 0x20000u;
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

__device__ static inline Pack128 zero_pack128_device() {
    Pack128 value;
    value.v4 = int32x4_t{0, 0, 0, 0};
    return value;
}

__device__ static inline Pack128 buffer_load_fp8_b128_rowptr_device(
    const int64_t* row_x_ptrs,
    int logical_row,
    int row_byte_offset) {
    const int64_t row_addr = row_x_ptrs[logical_row];
    if (__builtin_expect(row_addr <= 0, 0))
        return zero_pack128_device();
    const auto* row_ptr = reinterpret_cast<const uint8_t*>(row_addr);
    Pack128 value;
    value.v4 = *reinterpret_cast<const int32x4_t*>(row_ptr + row_byte_offset);
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

__device__ static inline void v3_k1_ll_grid_barrier_device(
    int32_t* barrier,
    int expected_blocks) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        volatile int32_t* counter = reinterpret_cast<volatile int32_t*>(barrier);
        volatile int32_t* sense_ptr =
            reinterpret_cast<volatile int32_t*>(barrier + 1);
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
}

__device__ static inline void v3_k1_ll_grid_barrier_init_device(
    int32_t* barrier,
    int epoch) {
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        barrier[0] = 0;
        __threadfence();
        barrier[1] = epoch;
    }
    if (threadIdx.x == 0) {
        volatile int32_t* phase =
            reinterpret_cast<volatile int32_t*>(barrier + 1);
        while (*phase != epoch) {
        }
    }
    __syncthreads();
}

__device__ static inline int v3_k1_clamp_num_tokens_device(
    int value,
    int num_max_tokens_per_rank) {
    if (value < 0)
        value = 0;
    if (value > num_max_tokens_per_rank)
        value = num_max_tokens_per_rank;
    return value;
}

__device__ static inline int v3_k1_section_num_tokens_device(
    const int32_t* runtime_num_tokens_ptr,
    int num_max_tokens_per_rank) {
    return v3_k1_clamp_num_tokens_device(
        runtime_num_tokens_ptr[0], num_max_tokens_per_rank);
}

template <int kExperts,
          int kK,
          int kBlockM,
          bool kParallelStageCopy = false>
__device__ static inline int32_t* v3_k1_build_ll_stage_device(
    const uint8_t* x,
    const float* x_scale,
    uint8_t* local_sym_buffer,
    int32_t* route_scratch_i32,
    int32_t* grid_barrier,
    int barrier_epoch,
    int rank_idx,
    int num_ranks,
    int num_global_experts,
    int num_max_tokens_per_rank,
    int num_topk,
    int runtime_num_tokens,
    int m_per_expert,
    float* route_weights_out,
    int32_t* row_expert_out,
    int32_t* output_index,
    int64_t* row_combine_ptrs_out,
    uint8_t* local_topk_mask,
    int32_t* tail_tokens,
    int32_t* cumulative_local_expert_recv_stats) {
    int32_t* symm_counts = route_scratch_i32;
    int64_t* symm_src_x_ptrs =
        reinterpret_cast<int64_t*>(route_scratch_i32 + kExperts);
    const int row_capacity = kExperts * m_per_expert;
    const int max_output_routes_per_rank = num_max_tokens_per_rank * num_topk;
    const int total_output_routes = num_ranks * max_output_routes_per_rank;
    const int tid = static_cast<int>(threadIdx.x);
    const int grid_threads = static_cast<int>(gridDim.x) *
                             static_cast<int>(blockDim.x);
    const int global_tid =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
    uint8_t* staged_x = const_cast<uint8_t*>(x);
    float* staged_x_scale = const_cast<float*>(x_scale);

    v3_k1_ll_grid_barrier_init_device(grid_barrier, barrier_epoch);

    for (int idx = global_tid; idx < kExperts; idx += grid_threads) {
        symm_counts[idx] = 0;
    }
    for (int idx = global_tid; idx < row_capacity; idx += grid_threads) {
        const int expert = idx / m_per_expert;
        symm_src_x_ptrs[idx] = 0;
        if (route_weights_out != nullptr)
            route_weights_out[idx] = 0.0f;
        if (row_expert_out != nullptr)
            row_expert_out[idx] = expert;
        if (row_combine_ptrs_out != nullptr)
            row_combine_ptrs_out[idx] = 0;
        staged_x_scale[idx] = 1.0e-4f / 448.0f;
    }
    if (output_index != nullptr) {
        for (int route_linear = global_tid;
             route_linear < total_output_routes;
             route_linear += grid_threads) {
            output_index[route_linear] = -1;
        }
    }
    v3_k1_ll_grid_barrier_device(grid_barrier, static_cast<int>(gridDim.x));

    uint8_t** peer_sym_buffers =
        deep_gemm::mega::dcu_peer_sym_buffer_ptrs(local_sym_buffer);
    const int local_experts = num_global_experts / num_ranks;
    const int first_expert = rank_idx * local_experts;
    const int last_expert = first_expert + local_experts;
    auto local_sections = deep_gemm::mega::get_sections(
        peer_sym_buffers[rank_idx], num_ranks, num_global_experts,
        num_max_tokens_per_rank, num_topk, kK);
    const int local_effective_tokens = v3_k1_section_num_tokens_device(
        local_sections.num_tokens, num_max_tokens_per_rank);
    const bool uniform_runtime_tokens =
        local_sections.uniform_num_tokens != nullptr &&
        local_sections.uniform_num_tokens[0] != 0;
    const int route_token_stride =
        uniform_runtime_tokens ? local_effective_tokens
                               : num_max_tokens_per_rank;
    const int route_scan_routes_per_rank = route_token_stride * num_topk;
    const int total_scan_routes = num_ranks * route_scan_routes_per_rank;

    if (local_topk_mask != nullptr || tail_tokens != nullptr) {
        for (int token_idx = global_tid; token_idx < num_max_tokens_per_rank;
             token_idx += grid_threads) {
            uint8_t mask = 0;
            if (token_idx < local_effective_tokens) {
                for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {
                    const int64_t route_offset =
                        static_cast<int64_t>(token_idx) * num_topk + topk_slot;
                    const int64_t expert = local_sections.topk_idx[route_offset];
                    const float weight =
                        local_sections.topk_weights[route_offset];
                    if (expert >= 0 && expert < num_global_experts &&
                        weight != 0.0f) {
                        mask |= static_cast<uint8_t>(1u << topk_slot);
                    }
                }
            }
            if (local_topk_mask != nullptr)
                local_topk_mask[token_idx] = mask;
            if (tail_tokens != nullptr)
                tail_tokens[token_idx] = token_idx;
        }
        v3_k1_ll_grid_barrier_device(grid_barrier, static_cast<int>(gridDim.x));
    }

    for (int route_scan_linear = global_tid; route_scan_linear < total_scan_routes;
         route_scan_linear += grid_threads) {
            const int source_rank =
                route_scan_linear / route_scan_routes_per_rank;
            const int local_route =
                route_scan_linear - source_rank * route_scan_routes_per_rank;
            const int token_idx = local_route / num_topk;
            const int topk_slot = local_route - token_idx * num_topk;
            auto sections = deep_gemm::mega::get_sections(
                peer_sym_buffers[source_rank], num_ranks, num_global_experts,
                num_max_tokens_per_rank, num_topk, kK);
            const int effective_tokens =
                uniform_runtime_tokens
                    ? route_token_stride
                    : v3_k1_section_num_tokens_device(
                          sections.num_tokens, num_max_tokens_per_rank);
            if (token_idx >= effective_tokens)
                continue;
            const int64_t expert =
                sections.topk_idx[static_cast<int64_t>(token_idx) * num_topk +
                                  topk_slot];
            if (expert < first_expert || expert >= last_expert)
                continue;
            const float route_weight =
                sections
                    .topk_weights[static_cast<int64_t>(token_idx) * num_topk +
                                  topk_slot];
            if (route_weight == 0.0f)
                continue;
            const int local_expert = static_cast<int>(expert) - first_expert;
            const int row_in_expert =
                atomicAdd(symm_counts + local_expert, 1);
            if (row_in_expert >= m_per_expert)
                continue;
            const int row = local_expert * m_per_expert + row_in_expert;
            symm_src_x_ptrs[row] =
                static_cast<int64_t>(reinterpret_cast<uintptr_t>(
                    sections.x + static_cast<int64_t>(token_idx) * kK));
            staged_x_scale[row] = sections.x_sf[token_idx];
            if (route_weights_out != nullptr)
                route_weights_out[row] = route_weight;
            if (row_expert_out != nullptr)
                row_expert_out[row] = local_expert;
            if (output_index != nullptr) {
                const int output_route_linear =
                    source_rank * num_max_tokens_per_rank * num_topk +
                    token_idx * num_topk + topk_slot;
                output_index[output_route_linear] = row;
            }
            if (row_combine_ptrs_out != nullptr) {
                const int64_t partial_row =
                    static_cast<int64_t>(topk_slot) *
                        num_max_tokens_per_rank +
                    token_idx;
                row_combine_ptrs_out[row] =
                    static_cast<int64_t>(reinterpret_cast<uintptr_t>(
                        sections.combine + partial_row * kK));
            }
        }
    v3_k1_ll_grid_barrier_device(grid_barrier, static_cast<int>(gridDim.x));

    if (cumulative_local_expert_recv_stats != nullptr) {
        for (int expert = global_tid; expert < kExperts; expert += grid_threads) {
            const int count = symm_counts[expert] > m_per_expert
                                  ? m_per_expert
                                  : symm_counts[expert];
            atomicAdd(cumulative_local_expert_recv_stats + expert,
                      count);
        }
    }

    constexpr int kStageVecBytes = 16;
    constexpr int kStageVecsPerRow = kK / kStageVecBytes;
    auto* staged_vecs = reinterpret_cast<int32x4_t*>(staged_x);
    const int32x4_t zero_vec{0, 0, 0, 0};
    if constexpr (kParallelStageCopy) {
        const int copy_ctas_per_expert =
            static_cast<int>(gridDim.x) >= kExperts
                ? static_cast<int>(gridDim.x) / kExperts
                : 1;
        if (static_cast<int>(gridDim.x) >= kExperts) {
            const int expert = static_cast<int>(blockIdx.x) % kExperts;
            const int expert_cta = static_cast<int>(blockIdx.x) / kExperts;
            if (expert_cta < copy_ctas_per_expert) {
                const int expert_count = symm_counts[expert] > m_per_expert
                                             ? m_per_expert
                                             : symm_counts[expert];
                const int stage_rows =
                    ((expert_count + kBlockM - 1) / kBlockM) * kBlockM;
                const int64_t expert_vecs =
                    static_cast<int64_t>(stage_rows) * kStageVecsPerRow;
                const int64_t stride =
                    static_cast<int64_t>(copy_ctas_per_expert) * blockDim.x;
                for (int64_t expert_vec =
                         static_cast<int64_t>(expert_cta) * blockDim.x + tid;
                     expert_vec < expert_vecs; expert_vec += stride) {
                    const int row_in_expert =
                        static_cast<int>(expert_vec / kStageVecsPerRow);
                    const int vec_col =
                        static_cast<int>(
                            expert_vec -
                            static_cast<int64_t>(row_in_expert) *
                                kStageVecsPerRow);
                    const int row = expert * m_per_expert + row_in_expert;
                    int32x4_t value = zero_vec;
                    if (row_in_expert < expert_count) {
                        const int64_t source_x_ptr = symm_src_x_ptrs[row];
                        if (source_x_ptr != 0) {
                            value = reinterpret_cast<const int32x4_t*>(
                                reinterpret_cast<const uint8_t*>(
                                    static_cast<uintptr_t>(
                                        source_x_ptr)))[vec_col];
                        }
                    }
                    staged_vecs[static_cast<int64_t>(row) *
                                    kStageVecsPerRow +
                                vec_col] = value;
                }
            }
        } else {
            for (int expert = static_cast<int>(blockIdx.x); expert < kExperts;
                 expert += static_cast<int>(gridDim.x)) {
                const int expert_count = symm_counts[expert] > m_per_expert
                                             ? m_per_expert
                                             : symm_counts[expert];
                const int stage_rows =
                    ((expert_count + kBlockM - 1) / kBlockM) * kBlockM;
                const int64_t expert_vecs =
                    static_cast<int64_t>(stage_rows) * kStageVecsPerRow;
                for (int64_t expert_vec = tid; expert_vec < expert_vecs;
                     expert_vec += blockDim.x) {
                    const int row_in_expert =
                        static_cast<int>(expert_vec / kStageVecsPerRow);
                    const int vec_col =
                        static_cast<int>(
                            expert_vec -
                            static_cast<int64_t>(row_in_expert) *
                                kStageVecsPerRow);
                    const int row = expert * m_per_expert + row_in_expert;
                    int32x4_t value = zero_vec;
                    if (row_in_expert < expert_count) {
                        const int64_t source_x_ptr = symm_src_x_ptrs[row];
                        if (source_x_ptr != 0) {
                            value = reinterpret_cast<const int32x4_t*>(
                                reinterpret_cast<const uint8_t*>(
                                    static_cast<uintptr_t>(
                                        source_x_ptr)))[vec_col];
                        }
                    }
                    staged_vecs[static_cast<int64_t>(row) *
                                    kStageVecsPerRow +
                                vec_col] = value;
                }
            }
        }
    } else {
        for (int expert = static_cast<int>(blockIdx.x); expert < kExperts;
             expert += static_cast<int>(gridDim.x)) {
            const int expert_count = symm_counts[expert] > m_per_expert
                                         ? m_per_expert
                                         : symm_counts[expert];
            const int stage_rows =
                ((expert_count + kBlockM - 1) / kBlockM) * kBlockM;
            const int64_t expert_vecs =
                static_cast<int64_t>(stage_rows) * kStageVecsPerRow;
            for (int64_t expert_vec = tid; expert_vec < expert_vecs;
                 expert_vec += blockDim.x) {
                const int row_in_expert =
                    static_cast<int>(expert_vec / kStageVecsPerRow);
                const int vec_col =
                    static_cast<int>(
                        expert_vec -
                        static_cast<int64_t>(row_in_expert) *
                            kStageVecsPerRow);
                const int row = expert * m_per_expert + row_in_expert;
                int32x4_t value = zero_vec;
                if (row_in_expert < expert_count) {
                    const int64_t source_x_ptr = symm_src_x_ptrs[row];
                    if (source_x_ptr != 0) {
                        value = reinterpret_cast<const int32x4_t*>(
                            reinterpret_cast<const uint8_t*>(
                                static_cast<uintptr_t>(source_x_ptr)))[vec_col];
                    }
                }
                staged_vecs[static_cast<int64_t>(row) * kStageVecsPerRow +
                            vec_col] = value;
            }
        }
    }
    __threadfence();
    v3_k1_ll_grid_barrier_device(grid_barrier, static_cast<int>(gridDim.x));
    return symm_counts;
}

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
          bool kParallelStageCopy = false>
__global__ __launch_bounds__(256, 1) void
V3_K1_LowLatencyMaskedGroupGemmKernel(
    hip_bfloat16* out,
    const uint8_t* x,
    const uint8_t* weight_packed,
    const float* x_scale,
    const float* w_scale,
    const int32_t* actual_m,
    int m_per_expert,
    uint8_t* local_sym_buffer = nullptr,
    int32_t* route_scratch_i32 = nullptr,
    int32_t* grid_barrier = nullptr,
    int barrier_epoch = 1,
    int rank_idx = 0,
    int num_ranks = 1,
    int num_global_experts = kExperts,
    int num_max_tokens_per_rank = 0,
    int num_topk = 0,
    int runtime_num_tokens = -1,
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

    const int32_t* gemm_m = actual_m;
    if constexpr (kUseSymmStage) {
        gemm_m =
            v3_k1_build_ll_stage_device<
                kExperts, kK, kBlockM, kParallelStageCopy>(
            x, x_scale, local_sym_buffer, route_scratch_i32, grid_barrier,
            barrier_epoch, rank_idx, num_ranks, num_global_experts,
            num_max_tokens_per_rank, num_topk, runtime_num_tokens,
            m_per_expert, route_weights_out, row_expert_out, output_index,
            row_combine_ptrs_out, local_topk_mask, tail_tokens,
            cumulative_local_expert_recv_stats);
    }

    int local_tokens = 0;
    if (lane < kExperts) {
        const int count = gemm_m[lane];
        local_tokens = count > m_per_expert ? m_per_expert : count;
    }

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
                input_scale[mr] =
                    x_scale[static_cast<int64_t>(expert) * m_per_expert +
                            tile_m * kBlockM + mr * 16 + ld_row];
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
                    auto* store_ptr = reinterpret_cast<bf16x4_t*>(
                        out_warp + store_lane_offset +
                        rep * kNumWarps * kMmaN);
                    store_ptr[0] = st;
                }
            }

            tile_id += kCUs;
        }
        last_expert_end += expert_tiles;
    }
    return;
}
