#pragma once

// Stage-owned V3 K3 LL pack5 group GEMM and tail-reduce core. V3 normal
// combine uses the separate ASM-pack5 path.

#include <hip/hip_bfloat16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <mega_moe_dcu/common.cuh>
#include <mega_moe_dcu/comm.cuh>

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using float32x2_t = float __attribute__((ext_vector_type(2)));
using float32x4_t = float __attribute__((ext_vector_type(4)));
using bf16x4_t = uint16_t __attribute__((ext_vector_type(4)));

static constexpr int kV3K3TailDoneCounterRingSlots = 16;
static constexpr int kV3K3TailPeerReadyOffset =
    2 * kV3K3TailDoneCounterRingSlots;

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
    (void)wg_n_tile;
    return stage_iter;
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
    const uint32_t global_offset1 = global_offset0 + 1u * 0x10000u;
    const uint32_t global_offset2 = global_offset0 + 2u * 0x10000u;
    const uint32_t global_offset3 = global_offset0 + 3u * 0x10000u;
    const uint32_t global_offset4 = global_offset0 + 4u * 0x10000u;
    const uint32_t global_offset5 = global_offset0 + 5u * 0x10000u;
    const uint32_t global_offset6 = global_offset0 + 6u * 0x10000u;
    const uint32_t global_offset7 = global_offset0 + 7u * 0x10000u;
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
    const uint32_t global_offset1 = global_offset0 + 0x10000u;
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

__device__ static inline int64_t buffer_load_i64_device(
    const int32x4_t resource,
    int global_byte_offset) {
    const int32x2_t value =
        llvm_amdgcn_raw_buffer_load_i32x2(resource, global_byte_offset, 0, 0);
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
    const uint64_t lo = static_cast<uint32_t>(value.x);
    const uint64_t hi = static_cast<uint32_t>(value.y);
    return static_cast<int64_t>(lo | (hi << 32));
}

__device__ static inline Pack128 zero_pack128_device() {
    Pack128 value;
    value.v4 = int32x4_t{0, 0, 0, 0};
    return value;
}

__device__ static inline void buffer_store_bf16_device(
    const int32x4_t resource,
    uint16_t value,
    int global_byte_offset) {
    llvm_amdgcn_raw_buffer_store_ui16(value, resource, global_byte_offset, 0, 0);
    return;
}

__device__ static inline void global_store_bf16_device(
    uint16_t* ptr,
    uint16_t value) {
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    const uint32_t data = static_cast<uint32_t>(value);
    asm volatile("global_store_short %0, %1, off\n\t"
                 :
                 : "v"(addr), "v"(data)
                 : "memory");
}

__device__ static inline void global_store_u32x2_device(
    uint16_t* ptr,
    uint32_t lo,
    uint32_t hi) {
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    const int32x2_t data{static_cast<int32_t>(lo), static_cast<int32_t>(hi)};
    asm volatile("global_store_dwordx2 %0, %1, off\n\t"
                 :
                 : "v"(addr), "v"(data)
                 : "memory");
}

__device__ static inline void global_store_u32x4_device(
    uint16_t* ptr,
    uint4 value) {
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    const int32x4_t data{
        static_cast<int32_t>(value.x),
        static_cast<int32_t>(value.y),
        static_cast<int32_t>(value.z),
        static_cast<int32_t>(value.w)};
    asm volatile("global_store_dwordx4 %0, %1, off\n\t"
                 :
                 : "v"(addr), "v"(data)
                 : "memory");
}

__device__ static inline int64_t global_load_i64_glc_device(
    const int64_t* ptr) {
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    int32x2_t value;
    asm volatile("global_load_dwordx2 %0, %1, off glc\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 : "=v"(value)
                 : "v"(addr)
                 : "memory");
    const uint64_t lo = static_cast<uint32_t>(value.x);
    const uint64_t hi = static_cast<uint32_t>(value.y);
    return static_cast<int64_t>(lo | (hi << 32));
}

__device__ static inline int global_load_i32_glc_slc_device(
    const volatile int* ptr) {
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    int value;
    asm volatile("global_load_dword %0, %1, off glc slc\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 : "=v"(value)
                 : "v"(addr)
                 : "memory");
    return value;
}

__device__ static inline uint4 global_load_uint4_device(const uint4* ptr) {
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    int32x4_t value;
    asm volatile("global_load_dwordx4 %0, %1, off\n\t"
                 "s_waitcnt vmcnt(0)\n\t"
                 : "=v"(value)
                 : "v"(addr)
                 : "memory");
    return uint4{
        static_cast<uint32_t>(value.x),
        static_cast<uint32_t>(value.y),
        static_cast<uint32_t>(value.z),
        static_cast<uint32_t>(value.w)};
}

__device__ static inline void lds_store_bf16_device(
    uint8_t* lds_bytes,
    int row,
    int hidden,
    uint16_t value) {
    auto* ptr = reinterpret_cast<uint16_t*>(
        lds_bytes + (row * 256 + hidden) * 2);
    ptr[0] = value;
}

__device__ static inline Pack128 buffer_load_fp8_b128_rowptr_device(
    const int64_t* row_ptrs,
    int logical_row,
    int row_byte_offset) {
    const int64_t row_addr = global_load_i64_glc_device(row_ptrs + logical_row);
    if (__builtin_expect(row_addr <= 0, 0))
        return zero_pack128_device();
    const auto* row_ptr = reinterpret_cast<const uint8_t*>(row_addr);
    return buffer_load_fp8_b128_pack_device(
        make_buffer_resource_device(row_ptr), row_byte_offset);
}

__device__ static inline Pack128 buffer_load_fp8_b128_active_row_device(
    const int32x4_t resource,
    const int64_t* row_combine_ptrs,
    int logical_row,
    int row_stride,
    int row_byte_offset) {
    if (__builtin_expect(global_load_i64_glc_device(row_combine_ptrs + logical_row) <= 0, 0))
        return zero_pack128_device();
    return buffer_load_fp8_b128_pack_device(
        resource,
        static_cast<int>(static_cast<int64_t>(logical_row) * row_stride +
                         row_byte_offset));
}

__device__ static inline Pack128 buffer_load_fp8_b128_active_row_buffer_device(
    const int32x4_t resource,
    const int32x4_t rowptr_resource,
    int logical_row,
    int row_stride,
    int row_byte_offset) {
    if (__builtin_expect(
            buffer_load_i64_device(rowptr_resource, logical_row * 8) <= 0, 0))
        return zero_pack128_device();
    return buffer_load_fp8_b128_pack_device(
        resource,
        static_cast<int>(static_cast<int64_t>(logical_row) * row_stride +
                         row_byte_offset));
}

__device__ static inline void store_bf16_rowptr_device(
    const int64_t* row_combine_ptrs,
    int row,
    int hidden,
    uint16_t value) {
    const int64_t row_addr = global_load_i64_glc_device(row_combine_ptrs + row);
    if (__builtin_expect(row_addr <= 0, 0))
        return;
    auto* row_ptr = reinterpret_cast<uint16_t*>(row_addr);
    global_store_bf16_device(row_ptr + hidden, value);
}

__device__ static inline void store_bf16_rowptr_value_dep_device(
    const int64_t* row_combine_ptrs,
    int row,
    int hidden,
    uint32_t value) {
    asm volatile("s_nop 0\n\t" : "+v"(value) : : "memory");
    store_bf16_rowptr_device(
        row_combine_ptrs, row, hidden, static_cast<uint16_t>(value));
}

__device__ static inline void store_bf16_rowptr_buffer_device(
    const int32x4_t rowptr_resource,
    int row,
    int hidden,
    uint16_t value) {
    const int64_t row_addr = buffer_load_i64_device(rowptr_resource, row * 8);
    if (__builtin_expect(row_addr <= 0, 0))
        return;
    auto* row_ptr = reinterpret_cast<uint16_t*>(row_addr);
    global_store_bf16_device(row_ptr + hidden, value);
}

__device__ static inline void store_bf16_rowptr_value_dep_buffer_device(
    const int32x4_t rowptr_resource,
    int row,
    int hidden,
    uint32_t value) {
    asm volatile("s_nop 0\n\t" : "+v"(value) : : "memory");
    store_bf16_rowptr_buffer_device(
        rowptr_resource, row, hidden, static_cast<uint16_t>(value));
}

__device__ static inline int64_t i32x2_to_i64_device(int32x2_t value) {
    const uint64_t lo = static_cast<uint32_t>(value.x);
    const uint64_t hi = static_cast<uint32_t>(value.y);
    return static_cast<int64_t>(lo | (hi << 32));
}

__device__ static inline void store_bf16_rowptr4_buffer_device(
    const int32x4_t rowptr_resource,
    int row_base,
    int hidden,
    uint16_t value0,
    uint16_t value4,
    uint16_t value8,
    uint16_t value12) {
    const int32x2_t addr0 =
        llvm_amdgcn_raw_buffer_load_i32x2(rowptr_resource, (row_base + 0) * 8, 0, 0);
    const int32x2_t addr4 =
        llvm_amdgcn_raw_buffer_load_i32x2(rowptr_resource, (row_base + 4) * 8, 0, 0);
    const int32x2_t addr8 =
        llvm_amdgcn_raw_buffer_load_i32x2(rowptr_resource, (row_base + 8) * 8, 0, 0);
    const int32x2_t addr12 =
        llvm_amdgcn_raw_buffer_load_i32x2(rowptr_resource, (row_base + 12) * 8, 0, 0);
    asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");

    const int64_t row_addr0 = i32x2_to_i64_device(addr0);
    const int64_t row_addr4 = i32x2_to_i64_device(addr4);
    const int64_t row_addr8 = i32x2_to_i64_device(addr8);
    const int64_t row_addr12 = i32x2_to_i64_device(addr12);
    if (__builtin_expect(row_addr0 > 0, 1))
        global_store_bf16_device(reinterpret_cast<uint16_t*>(row_addr0) + hidden,
                                 value0);
    if (__builtin_expect(row_addr4 > 0, 1))
        global_store_bf16_device(reinterpret_cast<uint16_t*>(row_addr4) + hidden,
                                 value4);
    if (__builtin_expect(row_addr8 > 0, 1))
        global_store_bf16_device(reinterpret_cast<uint16_t*>(row_addr8) + hidden,
                                 value8);
    if (__builtin_expect(row_addr12 > 0, 1))
        global_store_bf16_device(reinterpret_cast<uint16_t*>(row_addr12) + hidden,
                                 value12);
}

__device__ static inline void store_bf16x4_rowaddr_device(
    int64_t row_addr,
    int hidden,
    bf16x4_t values) {
    auto* row_ptr = reinterpret_cast<uint16_t*>(row_addr);
    const uint32_t lo = static_cast<uint32_t>(values[0]) |
                        (static_cast<uint32_t>(values[1]) << 16);
    const uint32_t hi = static_cast<uint32_t>(values[2]) |
                        (static_cast<uint32_t>(values[3]) << 16);
    global_store_u32x2_device(row_ptr + hidden, lo, hi);
}

__device__ static inline void store_bf16x4_rowptr_device(
    const int64_t* row_combine_ptrs,
    int row,
    int hidden,
    bf16x4_t values) {
    const int64_t row_addr = global_load_i64_glc_device(row_combine_ptrs + row);
    if (__builtin_expect(row_addr <= 0, 0))
        return;
    store_bf16x4_rowaddr_device(row_addr, hidden, values);
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

__device__ static inline void invalidate_l1_device() {
    asm volatile("buffer_wbinvl1_vol\n" ::: "memory");
}

__device__ static inline int load_signal_system_acquire_device(
    const volatile int* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ static inline void signal_generation_max_system_device(
    int* address,
    int value) {
    int old = 0;
    while (value > old) {
        const int assumed = old;
        old = atomicCAS_system(address, assumed, value);
        if (old == assumed)
            break;
    }
}

__device__ static inline void v3_k3_tail_signal_peers_device(
    const int64_t* signal_addrs,
    int num_ranks,
    int signal_value) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        for (int rank = 0; rank < num_ranks; ++rank) {
            const int64_t addr = signal_addrs == nullptr ? 0 : signal_addrs[rank];
            if (addr != 0) {
                signal_generation_max_system_device(
                    reinterpret_cast<int*>(addr), signal_value);
            }
        }
    }
}

__device__ static inline void v3_k3_tail_wait_peer_signals_device(
    const int64_t* signal_addrs,
    int num_ranks,
    int signal_generation) {
    if (threadIdx.x == 0) {
        for (int rank = 0; rank < num_ranks; ++rank) {
            const int64_t addr = signal_addrs == nullptr ? 0 : signal_addrs[8 + rank];
            if (addr == 0)
                continue;
            auto* signal = reinterpret_cast<volatile int*>(addr);
            const auto start_time = clock64();
            while (load_signal_system_acquire_device(signal) < signal_generation) {
                invalidate_l1_device();
                if (clock64() - start_time > deep_gemm::mega::kBarrierTimeoutCycles) {
                    abort();
                }
            }
        }
        invalidate_l1_device();
    }
    block_barrier_device();
    invalidate_l1_device();
}

__device__ static inline int v3_k3_tail_runtime_signal_generation_device(
    int signal_generation,
    const int32_t* signal_generation_ptr) {
    int runtime_signal_generation = signal_generation;
    if (signal_generation_ptr != nullptr) {
        runtime_signal_generation = signal_generation_ptr[0];
        if (runtime_signal_generation <= 0)
            runtime_signal_generation = 1;
    }
    return runtime_signal_generation;
}

__device__ static inline int v3_k3_tail_done_counter_slot_device(
    int runtime_signal_generation,
    const int32_t* signal_generation_ptr) {
    return signal_generation_ptr != nullptr
               ? (runtime_signal_generation &
                  (kV3K3TailDoneCounterRingSlots - 1))
               : 0;
}

__device__ static inline int* v3_k3_tail_peer_ready_counter_device(
    int32_t* done_counter,
    int slot) {
    return done_counter == nullptr ? nullptr
                                   : done_counter +
                                         kV3K3TailPeerReadyOffset + slot;
}

__device__ static inline void v3_k3_tail_publish_peer_ready_device(
    int* peer_ready_counter,
    int signal_generation) {
    if (threadIdx.x == 0) {
        __hip_atomic_store(peer_ready_counter, signal_generation,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    block_barrier_device();
}

__device__ static inline void v3_k3_tail_wait_peer_ready_device(
    int* peer_ready_counter,
    int signal_generation) {
    if (threadIdx.x == 0) {
        auto* ready = reinterpret_cast<volatile int*>(peer_ready_counter);
        const auto start_time = clock64();
        while (load_signal_system_acquire_device(ready) < signal_generation) {
            invalidate_l1_device();
            if (clock64() - start_time > deep_gemm::mega::kBarrierTimeoutCycles)
                abort();
        }
        invalidate_l1_device();
    }
    block_barrier_device();
    invalidate_l1_device();
}

__device__ static inline int v3_k3_tail_active_reduce_blocks_device(
    int num_tokens,
    const int32_t* runtime_num_tokens,
    int reduce_blocks) {
    if (reduce_blocks <= 64)
        return reduce_blocks;
    int effective_num_tokens = num_tokens;
    if (runtime_num_tokens != nullptr) {
        effective_num_tokens = runtime_num_tokens[0];
        if (effective_num_tokens < 0)
            effective_num_tokens = 0;
        if (effective_num_tokens > num_tokens)
            effective_num_tokens = num_tokens;
    }
    return effective_num_tokens <= 256 ? 64 : reduce_blocks;
}

__device__ static inline int v3_k3_tail_effective_tokens_device(
    int num_tokens,
    const int32_t* runtime_num_tokens) {
    int effective_num_tokens = num_tokens;
    if (runtime_num_tokens != nullptr) {
        effective_num_tokens = runtime_num_tokens[0];
        if (effective_num_tokens < 0)
            effective_num_tokens = 0;
        if (effective_num_tokens > num_tokens)
            effective_num_tokens = num_tokens;
    }
    return effective_num_tokens;
}

__device__ static inline void v3_k3_tail_reduce_worker_device(
    uint16_t* reduce_y,
    uint8_t* local_sym_buffer,
    int worker_idx,
    int worker_count,
    int num_ranks,
    int num_experts,
    int num_max_tokens_per_rank,
    int num_tokens,
    const int32_t* runtime_num_tokens,
    int num_topk,
    int hidden,
    int threads_per_worker = 768) {
    if (reduce_y == nullptr || local_sym_buffer == nullptr || worker_count <= 0)
        return;
    constexpr int kBf16PerVec = 8;
    constexpr int kTailReduceThreads = 768;
    const int reduce_threads =
        threads_per_worker > 0 ? threads_per_worker : kTailReduceThreads;
    const int reduce_tid = static_cast<int>(threadIdx.x);
    if (reduce_tid >= reduce_threads)
        return;
    const int effective_num_tokens =
        v3_k3_tail_effective_tokens_device(num_tokens, runtime_num_tokens);
    const int vecs_per_token = hidden / kBf16PerVec;
    const int64_t total_reduce_vecs =
        static_cast<int64_t>(effective_num_tokens) * vecs_per_token;
    const uint16_t* combine_base = deep_gemm::mega::get_sections(
        local_sym_buffer, num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk, hidden).combine;
    auto* y_vec = reinterpret_cast<uint4*>(reduce_y);
    for (int64_t task =
             static_cast<int64_t>(worker_idx) * reduce_threads + reduce_tid;
         task < total_reduce_vecs;
         task += static_cast<int64_t>(worker_count) * reduce_threads) {
        const int token_idx = static_cast<int>(task / vecs_per_token);
        const int vec_idx =
            static_cast<int>(task - static_cast<int64_t>(token_idx) * vecs_per_token);
        uint4 out;
#define V3_K3_TAIL_REDUCE_PAIR(FIELD)                                           \
        do {                                                                    \
            float sum_lo = 0.0f;                                                \
            float sum_hi = 0.0f;                                                \
            for (int topk_slot = 0; topk_slot < num_topk; ++topk_slot) {        \
                const int64_t partial_row =                                     \
                    static_cast<int64_t>(topk_slot) * num_max_tokens_per_rank + \
                    token_idx;                                                  \
                const auto packed =                                             \
                    global_load_uint4_device(                                   \
                        reinterpret_cast<const uint4*>(                         \
                            combine_base + partial_row * hidden) + vec_idx);    \
                const uint32_t word = packed.FIELD;                             \
                sum_lo += deep_gemm::mega::bf16_bits_to_float(                  \
                    static_cast<uint16_t>(word));                               \
                sum_hi += deep_gemm::mega::bf16_bits_to_float(                  \
                    static_cast<uint16_t>(word >> 16));                         \
            }                                                                   \
            out.FIELD = deep_gemm::mega::pack2_f32_to_bf16_bits(sum_lo, sum_hi);\
        } while (0)
        V3_K3_TAIL_REDUCE_PAIR(x);
        V3_K3_TAIL_REDUCE_PAIR(y);
        V3_K3_TAIL_REDUCE_PAIR(z);
        V3_K3_TAIL_REDUCE_PAIR(w);
#undef V3_K3_TAIL_REDUCE_PAIR
        y_vec[task] = out;
    }
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
          bool kUseFixedRows = false,
          bool kTailReduce = false>
__global__ __launch_bounds__(256, 1) void
V3_K3_LowLatencyMaskedGroupGemmKernel(
    hip_bfloat16* out,
    const uint8_t* x,
    const uint8_t* weight_packed,
    const float* x_scale,
    const float* w_scale,
    const int32_t* actual_m,
    int m_per_expert,
    const int64_t* row_combine_ptrs = nullptr,
    uint8_t* sym_buffer = nullptr,
    int32_t* done_counter = nullptr,
    const int64_t* signal_addrs = nullptr,
    uint16_t* reduce_y = nullptr,
    int num_ranks = 0,
    int num_experts = 0,
    int num_max_tokens_per_rank = 0,
    int num_tokens = 0,
    const int32_t* runtime_num_tokens = nullptr,
    int num_topk = 0,
    int signal_generation = 1,
    const int32_t* signal_generation_ptr = nullptr,
    int done_target = 0,
    int reduce_blocks = 0) {
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
    if constexpr (kTailReduce) {
        const int active_reduce_blocks =
            v3_k3_tail_active_reduce_blocks_device(
                num_tokens, runtime_num_tokens, reduce_blocks);
        if (static_cast<int>(blockIdx.x) >= kCUs) {
            const int reducer_idx = static_cast<int>(blockIdx.x) - kCUs;
            if (reducer_idx < active_reduce_blocks) {
                const int runtime_signal_generation =
                    v3_k3_tail_runtime_signal_generation_device(
                        signal_generation, signal_generation_ptr);
                const int slot = v3_k3_tail_done_counter_slot_device(
                    runtime_signal_generation, signal_generation_ptr);
                int* peer_ready_counter =
                    v3_k3_tail_peer_ready_counter_device(done_counter, slot);
                if (peer_ready_counter != nullptr &&
                    active_reduce_blocks > 1) {
                    if (reducer_idx == 0) {
                        v3_k3_tail_wait_peer_signals_device(
                            signal_addrs, num_ranks,
                            runtime_signal_generation);
                        v3_k3_tail_publish_peer_ready_device(
                            peer_ready_counter,
                            runtime_signal_generation);
                    } else {
                        v3_k3_tail_wait_peer_ready_device(
                            peer_ready_counter,
                            runtime_signal_generation);
                    }
                } else {
                    v3_k3_tail_wait_peer_signals_device(
                        signal_addrs, num_ranks, runtime_signal_generation);
                }
                v3_k3_tail_reduce_worker_device(
                    reduce_y, sym_buffer, reducer_idx, active_reduce_blocks,
                    num_ranks, num_experts, num_max_tokens_per_rank, num_tokens,
                    runtime_num_tokens, num_topk, kN,
                    static_cast<int>(blockDim.x));
            }
            return;
        }
    }

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

    int local_tokens = m_per_expert;
    if constexpr (!kUseFixedRows) {
        local_tokens = 0;
        if (lane < kExperts) {
            const int count = actual_m[lane];
            local_tokens = count > m_per_expert ? m_per_expert : count;
        }
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
            int64_t row_addr_prefetch[kMRepeats] = {};
            if (row_combine_ptrs != nullptr) {
#pragma unroll
                for (int mr = 0; mr < kMRepeats; ++mr) {
                    const int row_in_expert =
                        tile_m * kBlockM + mr * 16 + ld_row;
                    if constexpr (kMaskTinyStore) {
                        if (row_in_expert >= cur_tokens)
                            continue;
                    }
                    const int logical_row =
                        expert * m_per_expert + row_in_expert;
                    row_addr_prefetch[mr] = global_load_i64_glc_device(
                        row_combine_ptrs + logical_row);
                }
            }
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
                const int row_in_expert =
                    tile_m * kBlockM + mr * 16 + ld_row;
                if constexpr (kMaskTinyStore) {
                    if (row_in_expert >= cur_tokens)
                        continue;
                }
                int64_t row_addr = 0;
                if (row_combine_ptrs != nullptr) {
                    row_addr = row_addr_prefetch[mr];
                    if (__builtin_expect(row_addr <= 0, 0))
                        continue;
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
                    if (row_combine_ptrs != nullptr) {
                        const int hidden_base =
                            tile_n * kBlockN + warp_idx * kMmaN +
                            ld_col * 4 + rep * kNumWarps * kMmaN;
                        store_bf16x4_rowaddr_device(row_addr, hidden_base, st);
                    } else {
                        hip_bfloat16* out_warp =
                            expert_out +
                            (static_cast<int64_t>(tile_m) * kBlockM + mr * 16) *
                                kN +
                            tile_n * kBlockN + warp_idx * kMmaN;
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
    if constexpr (kTailReduce) {
        __shared__ int tail_signal_generation_shared;
        __shared__ int* tail_done_counter_shared;
        if (threadIdx.x == 0) {
            const int runtime_signal_generation =
                v3_k3_tail_runtime_signal_generation_device(
                    signal_generation, signal_generation_ptr);
            const int slot = v3_k3_tail_done_counter_slot_device(
                runtime_signal_generation, signal_generation_ptr);
            tail_signal_generation_shared = runtime_signal_generation;
            tail_done_counter_shared = done_counter + 2 * slot;
        }
        block_barrier_device();
        int* runtime_done_counter = tail_done_counter_shared;
        const int runtime_signal_generation = tail_signal_generation_shared;
        wait_vmem_lds_store_device();
        block_barrier_device();
        __threadfence_system();
        const int completion_owner_id = static_cast<int>(blockIdx.x) + 1;
        if (threadIdx.x == 0) {
            const int old = atomicAdd_system(runtime_done_counter, 1);
            if (old + 1 == done_target) {
                __threadfence_system();
                __hip_atomic_store(runtime_done_counter + 1,
                                   completion_owner_id,
                                   __ATOMIC_RELEASE,
                                   __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
        block_barrier_device();
        const int should_signal =
            deep_gemm::mega::load_signal_system(
                reinterpret_cast<volatile int*>(runtime_done_counter + 1)) ==
            completion_owner_id;
        if (should_signal) {
            v3_k3_tail_signal_peers_device(
                signal_addrs, num_ranks, runtime_signal_generation);
            if (reduce_blocks <= 0) {
                v3_k3_tail_wait_peer_signals_device(
                    signal_addrs, num_ranks, runtime_signal_generation);
                v3_k3_tail_reduce_worker_device(
                    reduce_y, sym_buffer, 0, 1,
                    num_ranks, num_experts, num_max_tokens_per_rank,
                    num_tokens, runtime_num_tokens, num_topk, kN,
                    static_cast<int>(blockDim.x));
            }
        }
    }
    return;
}
