#pragma once

#include <deep_gemm/common/mega_moe_dcu.cuh>

namespace deep_gemm::mega {

template <
    int kTileN_,
    int kKStageBytes_,
    int kLdsAStagePadBytes_ = 0>
struct DcuMmacTileShape {
    static constexpr int kTileN = kTileN_;
    static constexpr int kKStageBytes = kKStageBytes_;
    static constexpr int kLdsAStagePadBytes = kLdsAStagePadBytes_;
    static_assert(kTileN > 0 && kKStageBytes > 0);
    static_assert(kLdsAStagePadBytes >= 0);
    static_assert(kLdsAStagePadBytes % 16 == 0);
};

struct DcuL2BStageGlobal {
    static constexpr bool kUseLds = false;
};

struct DcuL2BStageLds {
    static constexpr bool kUseLds = true;
};

template <typename TileShape_>
struct DcuL1MmacTilePolicy {
    using TileShape = TileShape_;
    using L2BStagePolicy = DcuL2BStageGlobal;
    static constexpr int kStageBuffers = 1;
};

template <typename TileShape_, typename L2BStagePolicy_ = DcuL2BStageGlobal>
struct DcuL2MmacTilePolicy {
    using TileShape = TileShape_;
    using L2BStagePolicy = L2BStagePolicy_;
    static constexpr int kStageBuffers = 2;
};

template <typename TilePolicy_, int kThreads_>
struct DcuResolvedMmacTileConfig {
    using TilePolicy = TilePolicy_;
    using TileShape = typename TilePolicy::TileShape;
    using L2BStagePolicy = typename TilePolicy::L2BStagePolicy;

    static constexpr int kThreads = kThreads_;
    static constexpr int kWaveSize = 64;
    static constexpr int kMmacTileN = kDcuMmacTileN;
    static_assert(kThreads % kWaveSize == 0);
    static constexpr int kWaveCount = kThreads / kWaveSize;

    static constexpr int kTileM = kDcuMmacTileM;
    static constexpr int kTileN = TileShape::kTileN;
    static constexpr int kKStageBytes = TileShape::kKStageBytes;
    static constexpr int kStageBuffers = TilePolicy::kStageBuffers;
    static constexpr bool kDoubleBuffer = kStageBuffers == 2;
    static_assert(kThreads % kTileM == 0);
    static_assert(kTileN % (kWaveCount * kMmacTileN) == 0);
    static constexpr int kNChunksPerWave = kTileN / (kWaveCount * kMmacTileN);

    static constexpr int kLdsAStagePadBytes = TileShape::kLdsAStagePadBytes;
    static constexpr int kLdsAStageStrideBytes = kKStageBytes + kLdsAStagePadBytes;
    static_assert(kLdsAStageStrideBytes % 16 == 0);
    static constexpr int kLdsAStageVecs = kTileM * kLdsAStageStrideBytes / 16;
    static constexpr int kL1ChunkAmaxGroupsPerRow = kThreads / kTileM;
    static constexpr int kL1ChunkLocalQuantVecs =
        (kTileM * kTileN * static_cast<int>(sizeof(uint16_t)) +
         kTileM * kL1ChunkAmaxGroupsPerRow * static_cast<int>(sizeof(float)) + 15) / 16;
    static constexpr int kL2LdsBStageRows = L2BStagePolicy::kUseLds ? kTileN : 0;
    static constexpr int kL2LdsBStageVecs = kL2LdsBStageRows * kKStageBytes / 16;
    static constexpr int kLdsBStageVecs = kL2LdsBStageVecs;
    static constexpr int kLdsStageVecs = kLdsAStageVecs + kLdsBStageVecs;
    static constexpr int kLdsStageTotalVecs = kLdsStageVecs * kStageBuffers;
};

using DcuL1N256Pad64TilePolicy = DcuL1MmacTilePolicy<DcuMmacTileShape<256, 1024, 64>>;
using DcuL2N512TilePolicy = DcuL2MmacTilePolicy<DcuMmacTileShape<512, 512>>;

using uint32x4_t = uint32_t __attribute__((ext_vector_type(4)));
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));

__device__ static inline uint32x4_t make_dcu_buffer_resource(const uint8_t* ptr) {
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    uint32x4_t resource{0, 0, 0x80000000u, 0x00020000u};
    resource[0] = static_cast<uint32_t>(addr);
    resource[1] = static_cast<uint32_t>(addr >> 32);
    return resource;
}

__device__ static inline void raw_buffer_load_lds_16b(
    const uint32x4_t resource,
    uint8_t* lds_base,
    const int lds_byte_offset,
    const int global_byte_offset) {
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + lds_byte_offset);
    auto* lds_ptr = (__attribute__((address_space(3))) int*)lds_addr;
    __builtin_amdgcn_raw_buffer_load_lds(
        resource, lds_ptr, 16, global_byte_offset, 0, 0, 0);
}

__device__ static inline int32x4_t load_fp8_lds_b128_nowait(
    const uint8_t* lds_base,
    const int byte_offset) {
    const int pair_offset = byte_offset & ~15;
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + pair_offset);
    int32x4_t vec;
    asm volatile(
        "ds_read_b128 %0, %1\n"
        : "=v"(vec)
        : "v"(static_cast<uint32_t>(lds_addr))
        : "memory");
    return vec;
}

__device__ static inline dcu::int32x2_t load_fp8_lds_b64_nowait(
    const uint8_t* lds_base,
    const int byte_offset) {
    const int pair_offset = byte_offset & ~7;
    const uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_base + pair_offset);
    dcu::int32x2_t vec;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        : "=v"(vec)
        : "v"(static_cast<uint32_t>(lds_addr))
        : "memory");
    return vec;
}

__device__ static inline dcu::int32x2_t select_fp8_pair_from_b128(
    const int32x4_t vec,
    const int byte_offset) {
    dcu::int32x2_t out;
    if ((byte_offset & 8) == 0) {
        out.x = vec.x;
        out.y = vec.y;
    } else {
        out.x = vec.z;
        out.y = vec.w;
    }
    return out;
}

__device__ static inline void wait_lds_reads() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

__device__ static inline void dcu_vec4_zero(dcu::float32x4_t& value) {
    value = dcu::float32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ static inline void dcu_vec4_accum_scaled(
    dcu::float32x4_t& dst,
    const dcu::float32x4_t src,
    const float scale) {
    dst.x += src.x * scale;
    dst.y += src.y * scale;
    dst.z += src.z * scale;
    dst.w += src.w * scale;
}

__device__ static inline void quantize_l1_chunk_group_from_lds(
    const uint16_t* act_chunk_bf16,
    uint8_t* act_fp8,
    const int row,
    const int chunk_cols,
    const int intermediate_hidden,
    const int inter_start,
    const int inter_base,
    const int c_n_base,
    const float inv_scale) {
    float values[4];
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int n = c_n_base + 4 * slot;
        const int inter_idx = inter_base + n;
        values[slot] = inter_idx < intermediate_hidden
            ? bf16_bits_to_float(
                  act_chunk_bf16[row * chunk_cols + inter_idx - inter_start]) * inv_scale
            : 0.0f;
    }
    const uint32_t packed = static_cast<uint32_t>(dcu::pack4_f32_to_fp8(values));
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
        const int n = c_n_base + 4 * slot;
        const int inter_idx = inter_base + n;
        if (inter_idx < intermediate_hidden)
            act_fp8[row * intermediate_hidden + inter_idx] =
                static_cast<uint8_t>((packed >> (8 * slot)) & 0xffu);
    }
}

#if defined(__gfx938__)
__device__ static inline void fp8_mmac8_buffer_asm(
    dcu::float32x4_t& acc0,
    dcu::float32x4_t& acc1,
    dcu::float32x4_t& acc2,
    dcu::float32x4_t& acc3,
    dcu::float32x4_t& acc4,
    dcu::float32x4_t& acc5,
    dcu::float32x4_t& acc6,
    dcu::float32x4_t& acc7,
    const dcu::int32x2_t a_vec,
    const uint32x4_t weight_resource,
    const uint32_t w_offset0,
    const uint32_t w_offset1,
    const uint32_t w_offset2,
    const uint32_t w_offset3,
    const uint32_t w_offset4,
    const uint32_t w_offset5,
    const uint32_t w_offset6,
    const uint32_t w_offset7) {
    dcu::int32x2_t w_vec0;
    dcu::int32x2_t w_vec1;
    dcu::int32x2_t w_vec2;
    dcu::int32x2_t w_vec3;
    dcu::int32x2_t w_vec4;
    dcu::int32x2_t w_vec5;
    dcu::int32x2_t w_vec6;
    dcu::int32x2_t w_vec7;
    asm volatile(
        "buffer_load_dwordx2 %8, %17, %25, 0 offen\n\t"
        "buffer_load_dwordx2 %9, %18, %25, 0 offen\n\t"
        "buffer_load_dwordx2 %10, %19, %25, 0 offen\n\t"
        "buffer_load_dwordx2 %11, %20, %25, 0 offen\n\t"
        "buffer_load_dwordx2 %12, %21, %25, 0 offen\n\t"
        "buffer_load_dwordx2 %13, %22, %25, 0 offen\n\t"
        "buffer_load_dwordx2 %14, %23, %25, 0 offen\n\t"
        "buffer_load_dwordx2 %15, %24, %25, 0 offen\n\t"
        "s_waitcnt vmcnt(7)\n\t"
        "v_mmac_f32_16x16x32_fp8_fp8 %0, %16, %8, %0\n\t"
        "s_waitcnt vmcnt(6)\n\t"
        "v_mmac_f32_16x16x32_fp8_fp8 %1, %16, %9, %1\n\t"
        "s_waitcnt vmcnt(5)\n\t"
        "v_mmac_f32_16x16x32_fp8_fp8 %2, %16, %10, %2\n\t"
        "s_waitcnt vmcnt(4)\n\t"
        "v_mmac_f32_16x16x32_fp8_fp8 %3, %16, %11, %3\n\t"
        "s_waitcnt vmcnt(3)\n\t"
        "v_mmac_f32_16x16x32_fp8_fp8 %4, %16, %12, %4\n\t"
        "s_waitcnt vmcnt(2)\n\t"
        "v_mmac_f32_16x16x32_fp8_fp8 %5, %16, %13, %5\n\t"
        "s_waitcnt vmcnt(1)\n\t"
        "v_mmac_f32_16x16x32_fp8_fp8 %6, %16, %14, %6\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "v_mmac_f32_16x16x32_fp8_fp8 %7, %16, %15, %7\n\t"
        : "+v"(acc0), "+v"(acc1), "+v"(acc2), "+v"(acc3),
          "+v"(acc4), "+v"(acc5), "+v"(acc6), "+v"(acc7),
          "=&v"(w_vec0), "=&v"(w_vec1), "=&v"(w_vec2), "=&v"(w_vec3),
          "=&v"(w_vec4), "=&v"(w_vec5), "=&v"(w_vec6), "=&v"(w_vec7)
        : "v"(a_vec),
          "v"(w_offset0), "v"(w_offset1), "v"(w_offset2), "v"(w_offset3),
          "v"(w_offset4), "v"(w_offset5), "v"(w_offset6), "v"(w_offset7),
          "s"(weight_resource)
        : "memory");
}
#endif

__device__ static inline void wait_mtile16_a_operand_to_lds() {
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

template <int kTileM, int kStageBytes, int kRowStrideBytes = kStageBytes>
__device__ static inline void stage_mtile_a_operand_to_lds_nowait_coop(
    const uint8_t* a,
    const int valid_rows,
    const int k_stage_start,
    const int k_total,
    const int stride_k,
    uint4* lds_a_vecs,
    const int coop_thread_id,
    const int coop_threads) {
    constexpr int kVecBytes = 16;
    static_assert(kStageBytes % kVecBytes == 0);
    static_assert(kRowStrideBytes % kVecBytes == 0);
    constexpr int kVecsPerStageRow = kStageBytes / kVecBytes;
    constexpr int kTotalVecs = kTileM * kVecsPerStageRow;
    auto* dst_bytes = reinterpret_cast<uint8_t*>(lds_a_vecs);
    const uint32x4_t a_resource = make_dcu_buffer_resource(a);

    for (int linear_vec = coop_thread_id; linear_vec < kTotalVecs; linear_vec += coop_threads) {
        const int row = linear_vec / kVecsPerStageRow;
        const int vec_idx = linear_vec - row * kVecsPerStageRow;
        const int k_idx = k_stage_start + vec_idx * kVecBytes;
        const int dst_byte_offset = row * kRowStrideBytes + vec_idx * kVecBytes;
        if (row < valid_rows && k_idx + kVecBytes <= k_total) {
            raw_buffer_load_lds_16b(
                a_resource, dst_bytes, dst_byte_offset, row * stride_k + k_idx);
        } else {
            uint4 value{0, 0, 0, 0};
            if (row < valid_rows && k_idx < k_total) {
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
#pragma unroll
                for (int i = 0; i < kVecBytes; ++i) {
                    if (k_idx + i < k_total)
                        bytes[i] = a[row * stride_k + k_idx + i];
                }
            }
            *reinterpret_cast<uint4*>(dst_bytes + dst_byte_offset) = value;
        }
    }
    asm volatile("" ::: "memory");
}

template <int kStageBytes, int kRowStrideBytes = kStageBytes>
__device__ static inline void stage_mtile16_a_operand_to_lds_nowait_coop(
    const uint8_t* a,
    const int valid_rows,
    const int k_stage_start,
    const int k_total,
    const int stride_k,
    uint4* lds_a_vecs,
    const int coop_thread_id,
    const int coop_threads) {
    stage_mtile_a_operand_to_lds_nowait_coop<16, kStageBytes, kRowStrideBytes>(
        a, valid_rows, k_stage_start, k_total, stride_k,
        lds_a_vecs, coop_thread_id, coop_threads);
}

template <int kStageBytes, int kRowStrideBytes = kStageBytes>
__device__ static inline void stage_mtile16_a_operand_to_lds_nowait(
    const uint8_t* a,
    const int valid_rows,
    const int k_stage_start,
    const int k_total,
    const int stride_k,
    uint4* lds_a_vecs) {
    stage_mtile16_a_operand_to_lds_nowait_coop<kStageBytes, kRowStrideBytes>(
        a, valid_rows, k_stage_start, k_total, stride_k, lds_a_vecs, threadIdx.x, blockDim.x);
}

template <int kStageBytes, int kRowStrideBytes = kStageBytes>
__device__ static inline void stage_mtile16_a_operand_to_lds(
    const uint8_t* a,
    const int valid_rows,
    const int k_stage_start,
    const int k_total,
    const int stride_k,
    uint4* lds_a_vecs) {
    stage_mtile16_a_operand_to_lds_nowait<kStageBytes, kRowStrideBytes>(
        a, valid_rows, k_stage_start, k_total, stride_k, lds_a_vecs);
    wait_mtile16_a_operand_to_lds();
}

template <typename TileConfig>
__device__ static inline void stage_l2_b_operand_to_lds_nowait_coop(
    const uint8_t* l2_weights,
    const int local_expert,
    const int hidden_start,
    const int k_stage_start,
    const int hidden,
    const int intermediate_hidden,
    uint4* lds_b_vecs,
    const int coop_thread_id,
    const int coop_threads) {
    constexpr int kVecBytes = 16;
    static_assert(TileConfig::kL2LdsBStageRows == TileConfig::kTileN);
    static_assert(TileConfig::kKStageBytes % kVecBytes == 0);
    constexpr int kVecsPerStageRow = TileConfig::kKStageBytes / kVecBytes;
    constexpr int kTotalVecs = TileConfig::kL2LdsBStageRows * kVecsPerStageRow;
    auto* dst_vecs = lds_b_vecs;

    for (int linear_vec = coop_thread_id; linear_vec < kTotalVecs; linear_vec += coop_threads) {
        const int row = linear_vec / kVecsPerStageRow;
        const int vec_idx = linear_vec - row * kVecsPerStageRow;
        const int h_idx = hidden_start + row;
        const int k_idx = k_stage_start + vec_idx * kVecBytes;
        uint4 value{0, 0, 0, 0};
        if (h_idx < hidden && k_idx + kVecBytes <= intermediate_hidden) {
            const int64_t src_offset = marlin_nt_kpack2_row_base_offset(
                local_expert, h_idx, hidden, intermediate_hidden) +
                marlin_nt_kpack2_k_offset(k_idx);
            value = *reinterpret_cast<const uint4*>(l2_weights + src_offset);
        }
        dst_vecs[linear_vec] = value;
    }
    asm volatile("" ::: "memory");
}

template <typename TileConfig, bool kChunkLocalQuant = false>
__device__ static inline void compute_route_mmac_mtile16_l1_chunk(
    const int local_expert,
    const int inter_start,
    const int valid_rows,
    const int64_t tile_meta_base,
    const int hidden,
    const int intermediate_hidden,
    const float activation_clamp,
    const uint8_t* tile_x_pool,
    const float* tile_route_weights,
    const float* tile_x_scales,
    const uint8_t* l1_weights, const float* l1_weights_sf,
    uint16_t* act_bf16,
    uint8_t* act_fp8,
    float* act_chunk_amax,
    const int num_inter_chunks,
    uint4* lds_a_stage) {
    constexpr int kTileM = TileConfig::kTileM;
    static_assert(TileConfig::kNChunksPerWave == 4);
    static_assert(!TileConfig::kDoubleBuffer);
    const int wave_id = static_cast<int>(threadIdx.x >> 6);
    const int lane = static_cast<int>(threadIdx.x & 63);
    const int lane_m = lane & 15;
    const int lane_n = lane & 15;
    const int lane_k = (lane >> 4) * 8;
    const int lane_group = lane >> 4;
    const int c_n_base = lane_group;
    const bool row_valid = lane_m < valid_rows;
    const int l1_rows = intermediate_hidden * 2;
    const float* l1_weights_sf_expert =
        l1_weights_sf + static_cast<int64_t>(local_expert) * l1_rows;
    const int inter_base0 =
        inter_start + wave_id * TileConfig::kNChunksPerWave * TileConfig::kMmacTileN;
    const int inter_base1 = inter_base0 + TileConfig::kMmacTileN;
    const int inter_base2 = inter_base0 + 2 * TileConfig::kMmacTileN;
    const int inter_base3 = inter_base0 + 3 * TileConfig::kMmacTileN;
    const bool inter_valid0 = inter_base0 + lane_n < intermediate_hidden;
    const bool inter_valid1 = inter_base1 + lane_n < intermediate_hidden;
    const bool inter_valid2 = inter_base2 + lane_n < intermediate_hidden;
    const bool inter_valid3 = inter_base3 + lane_n < intermediate_hidden;
    const bool do_clamp = isfinite(activation_clamp);
    constexpr int chunk_cols = TileConfig::kTileN;
    const int chunk_id = inter_start / chunk_cols;
    uint16_t* act_chunk_bf16 = reinterpret_cast<uint16_t*>(lds_a_stage);
    float* act_chunk_partials =
        reinterpret_cast<float*>(act_chunk_bf16 + kTileM * chunk_cols);

    const int64_t meta_idx = tile_meta_base + lane_m;
    float route_weight = 0.0f;
    float x_scale = 0.0f;
    if (row_valid) {
        route_weight = tile_route_weights[meta_idx];
        x_scale = tile_x_scales[meta_idx];
    }

    dcu::int32x2_t zero8;
    zero8.x = 0;
    zero8.y = 0;

    dcu::float32x4_t gate_acc0{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t up_acc0{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t gate_acc1{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t up_acc1{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t gate_acc2{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t up_acc2{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t gate_acc3{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t up_acc3{0.0f, 0.0f, 0.0f, 0.0f};
    const int gate_row0 = inter_base0 + lane_n;
    const int up_row0 = intermediate_hidden + gate_row0;
    const int gate_row1 = inter_base1 + lane_n;
    const int up_row1 = intermediate_hidden + gate_row1;
    const int gate_row2 = inter_base2 + lane_n;
    const int up_row2 = intermediate_hidden + gate_row2;
    const int gate_row3 = inter_base3 + lane_n;
    const int up_row3 = intermediate_hidden + gate_row3;
    const int64_t gate_base0 = marlin_nt_kpack2_row_base_offset(
        local_expert, gate_row0, l1_rows, hidden);
    const int64_t up_base0 = marlin_nt_kpack2_row_base_offset(
        local_expert, up_row0, l1_rows, hidden);
    const int64_t gate_base1 = marlin_nt_kpack2_row_base_offset(
        local_expert, gate_row1, l1_rows, hidden);
    const int64_t up_base1 = marlin_nt_kpack2_row_base_offset(
        local_expert, up_row1, l1_rows, hidden);
    const int64_t gate_base2 = marlin_nt_kpack2_row_base_offset(
        local_expert, gate_row2, l1_rows, hidden);
    const int64_t up_base2 = marlin_nt_kpack2_row_base_offset(
        local_expert, up_row2, l1_rows, hidden);
    const int64_t gate_base3 = marlin_nt_kpack2_row_base_offset(
        local_expert, gate_row3, l1_rows, hidden);
    const int64_t up_base3 = marlin_nt_kpack2_row_base_offset(
        local_expert, up_row3, l1_rows, hidden);
    const auto* lds_a = reinterpret_cast<const uint8_t*>(lds_a_stage);
    const uint8_t* subtile_x = tile_x_pool + tile_meta_base * static_cast<int64_t>(hidden);
    for (int k_stage = 0;
         k_stage < hidden;
        k_stage += TileConfig::kKStageBytes) {
        stage_mtile16_a_operand_to_lds<
            TileConfig::kKStageBytes,
            TileConfig::kLdsAStageStrideBytes>(
            subtile_x, valid_rows, k_stage, hidden, hidden, lds_a_stage);
        __syncthreads();
        const int k_stage_end = min(k_stage + TileConfig::kKStageBytes, hidden);
        for (int k_base = k_stage; k_base < k_stage_end; k_base += 32) {
            const int k_idx = k_base + lane_k;
            const int lds_k_idx = k_idx - k_stage;
            const int a_byte_offset =
                lane_m * TileConfig::kLdsAStageStrideBytes + lds_k_idx;
            dcu::int32x2_t a_vec = zero8;
            if (row_valid)
                a_vec = load_fp8_lds_b64_nowait(lds_a, a_byte_offset);
            const dcu::int32x2_t gate_vec0 = inter_valid0
                ? pack8_fp8_weight_marlin_row_base(l1_weights, gate_base0, k_idx)
                : zero8;
            const dcu::int32x2_t up_vec0 = inter_valid0
                ? pack8_fp8_weight_marlin_row_base(l1_weights, up_base0, k_idx)
                : zero8;
            const dcu::int32x2_t gate_vec1 = inter_valid1
                ? pack8_fp8_weight_marlin_row_base(l1_weights, gate_base1, k_idx)
                : zero8;
            const dcu::int32x2_t up_vec1 = inter_valid1
                ? pack8_fp8_weight_marlin_row_base(l1_weights, up_base1, k_idx)
                : zero8;
            const dcu::int32x2_t gate_vec2 = inter_valid2
                ? pack8_fp8_weight_marlin_row_base(l1_weights, gate_base2, k_idx)
                : zero8;
            const dcu::int32x2_t up_vec2 = inter_valid2
                ? pack8_fp8_weight_marlin_row_base(l1_weights, up_base2, k_idx)
                : zero8;
            const dcu::int32x2_t gate_vec3 = inter_valid3
                ? pack8_fp8_weight_marlin_row_base(l1_weights, gate_base3, k_idx)
                : zero8;
            const dcu::int32x2_t up_vec3 = inter_valid3
                ? pack8_fp8_weight_marlin_row_base(l1_weights, up_base3, k_idx)
                : zero8;
            wait_lds_reads();
            gate_acc0 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, gate_vec0, gate_acc0);
            up_acc0 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, up_vec0, up_acc0);
            gate_acc1 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, gate_vec1, gate_acc1);
            up_acc1 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, up_vec1, up_acc1);
            gate_acc2 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, gate_vec2, gate_acc2);
            up_acc2 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, up_vec2, up_acc2);
            gate_acc3 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, gate_vec3, gate_acc3);
            up_acc3 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, up_vec3, up_acc3);
        }
        __syncthreads();
    }

    float thread_act_amax = 0.0f;

    if (row_valid && inter_valid0) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int n = c_n_base + 4 * slot;
            const int inter_idx = inter_base0 + n;
            if (inter_idx < intermediate_hidden) {
                const float gate_scale = x_scale * l1_weights_sf_expert[inter_idx];
                const float up_scale = x_scale * l1_weights_sf_expert[intermediate_hidden + inter_idx];
                float gate = vec4_get(gate_acc0, slot) * gate_scale;
                float up = vec4_get(up_acc0, slot) * up_scale;
                if (do_clamp) {
                    gate = fminf(fmaxf(gate, -activation_clamp), activation_clamp);
                    up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                }
                const float silu = fast_silu(gate);
                const uint16_t act_bits = float_to_bf16_bits(silu * up * route_weight);
                thread_act_amax = fmaxf(thread_act_amax, fabsf(bf16_bits_to_float(act_bits)));
                if constexpr (kChunkLocalQuant)
                    act_chunk_bf16[lane_m * chunk_cols + inter_idx - inter_start] = act_bits;
                else
                    act_bf16[lane_m * intermediate_hidden + inter_idx] = act_bits;
            }
        }
    }
    if (row_valid && inter_valid1) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int n = c_n_base + 4 * slot;
            const int inter_idx = inter_base1 + n;
            if (inter_idx < intermediate_hidden) {
                const float gate_scale = x_scale * l1_weights_sf_expert[inter_idx];
                const float up_scale = x_scale * l1_weights_sf_expert[intermediate_hidden + inter_idx];
                float gate = vec4_get(gate_acc1, slot) * gate_scale;
                float up = vec4_get(up_acc1, slot) * up_scale;
                if (do_clamp) {
                    gate = fminf(fmaxf(gate, -activation_clamp), activation_clamp);
                    up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                }
                const float silu = fast_silu(gate);
                const uint16_t act_bits = float_to_bf16_bits(silu * up * route_weight);
                thread_act_amax = fmaxf(thread_act_amax, fabsf(bf16_bits_to_float(act_bits)));
                if constexpr (kChunkLocalQuant)
                    act_chunk_bf16[lane_m * chunk_cols + inter_idx - inter_start] = act_bits;
                else
                    act_bf16[lane_m * intermediate_hidden + inter_idx] = act_bits;
            }
        }
    }
    if (row_valid && inter_valid2) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int n = c_n_base + 4 * slot;
            const int inter_idx = inter_base2 + n;
            if (inter_idx < intermediate_hidden) {
                const float gate_scale = x_scale * l1_weights_sf_expert[inter_idx];
                const float up_scale = x_scale * l1_weights_sf_expert[intermediate_hidden + inter_idx];
                float gate = vec4_get(gate_acc2, slot) * gate_scale;
                float up = vec4_get(up_acc2, slot) * up_scale;
                if (do_clamp) {
                    gate = fminf(fmaxf(gate, -activation_clamp), activation_clamp);
                    up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                }
                const float silu = fast_silu(gate);
                const uint16_t act_bits = float_to_bf16_bits(silu * up * route_weight);
                thread_act_amax = fmaxf(thread_act_amax, fabsf(bf16_bits_to_float(act_bits)));
                if constexpr (kChunkLocalQuant)
                    act_chunk_bf16[lane_m * chunk_cols + inter_idx - inter_start] = act_bits;
                else
                    act_bf16[lane_m * intermediate_hidden + inter_idx] = act_bits;
            }
        }
    }
    if (row_valid && inter_valid3) {
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int n = c_n_base + 4 * slot;
            const int inter_idx = inter_base3 + n;
            if (inter_idx < intermediate_hidden) {
                const float gate_scale = x_scale * l1_weights_sf_expert[inter_idx];
                const float up_scale = x_scale * l1_weights_sf_expert[intermediate_hidden + inter_idx];
                float gate = vec4_get(gate_acc3, slot) * gate_scale;
                float up = vec4_get(up_acc3, slot) * up_scale;
                if (do_clamp) {
                    gate = fminf(fmaxf(gate, -activation_clamp), activation_clamp);
                    up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                }
                const float silu = fast_silu(gate);
                const uint16_t act_bits = float_to_bf16_bits(silu * up * route_weight);
                thread_act_amax = fmaxf(thread_act_amax, fabsf(bf16_bits_to_float(act_bits)));
                if constexpr (kChunkLocalQuant)
                    act_chunk_bf16[lane_m * chunk_cols + inter_idx - inter_start] = act_bits;
                else
                    act_bf16[lane_m * intermediate_hidden + inter_idx] = act_bits;
            }
        }
    }

    constexpr int kAmaxGroupsPerRow = TileConfig::kL1ChunkAmaxGroupsPerRow;
    if (row_valid) {
        const int amax_group = static_cast<int>(threadIdx.x >> kDcuMmacTileMLog2);
        act_chunk_partials[lane_m * kAmaxGroupsPerRow + amax_group] = thread_act_amax;
    }
    __syncthreads();
    if (threadIdx.x < kTileM) {
        const int row = threadIdx.x;
        float row_amax = 0.0f;
        if (row < valid_rows) {
#pragma unroll
            for (int group = 0; group < kAmaxGroupsPerRow; ++group)
                row_amax = fmaxf(row_amax, act_chunk_partials[row * kAmaxGroupsPerRow + group]);
            if constexpr (kChunkLocalQuant) {
                const float scale = fmaxf(row_amax / 448.0f, 1.0e-4f);
                act_chunk_partials[row] = 1.0f / scale;
                act_chunk_amax[(tile_meta_base + row) * num_inter_chunks + chunk_id] = scale;
            } else {
                act_chunk_amax[(tile_meta_base + row) * num_inter_chunks + chunk_id] = row_amax;
            }
        } else if constexpr (kChunkLocalQuant) {
            act_chunk_partials[row] = 1.0f;
        }
    }

    if constexpr (kChunkLocalQuant) {
        __syncthreads();
        if (row_valid) {
            const float inv_scale = act_chunk_partials[lane_m];
            quantize_l1_chunk_group_from_lds(
                act_chunk_bf16, act_fp8, lane_m, chunk_cols, intermediate_hidden,
                inter_start, inter_base0, c_n_base, inv_scale);
            quantize_l1_chunk_group_from_lds(
                act_chunk_bf16, act_fp8, lane_m, chunk_cols, intermediate_hidden,
                inter_start, inter_base1, c_n_base, inv_scale);
            quantize_l1_chunk_group_from_lds(
                act_chunk_bf16, act_fp8, lane_m, chunk_cols, intermediate_hidden,
                inter_start, inter_base2, c_n_base, inv_scale);
            quantize_l1_chunk_group_from_lds(
                act_chunk_bf16, act_fp8, lane_m, chunk_cols, intermediate_hidden,
                inter_start, inter_base3, c_n_base, inv_scale);
        }
    }
}

__device__ static inline void quant_bf16_act_channelwise_mtile16_global_with_chunk_amax(
    const uint16_t* act_bf16,
    uint8_t* act_fp8,
    float* act_scale,
    const float* act_chunk_amax,
    const int valid_rows,
    const int intermediate_hidden,
    const int num_inter_chunks) {
    const int wave_id = static_cast<int>(threadIdx.x >> 6);
    const int wave_count = static_cast<int>(blockDim.x >> 6);
    const int lane = static_cast<int>(threadIdx.x & 63);

    for (int row = wave_id; row < kDcuMmacTileM; row += wave_count) {
        const bool row_valid = row < valid_rows;
        float local_amax = 0.0f;
        if (row_valid) {
            for (int idx = lane; idx < num_inter_chunks; idx += 64)
                local_amax = fmaxf(local_amax, act_chunk_amax[row * num_inter_chunks + idx]);
        }

        for (int offset = 32; offset > 0; offset >>= 1)
            local_amax = fmaxf(local_amax, __shfl_down(local_amax, offset, 64));

        const float scale = row_valid ? fmaxf(__shfl(local_amax, 0, 64) / 448.0f, 1.0e-4f) : 1.0f;
        if (lane == 0)
            act_scale[row] = scale;

        if (row_valid) {
            const float inv_scale = 1.0f / scale;
            for (int i = lane * 4; i < intermediate_hidden; i += 64 * 4) {
                float values[4];
                const uint16_t* src = act_bf16 + row * intermediate_hidden + i;
                if (i + 3 < intermediate_hidden) {
                    const uint32_t lo = *reinterpret_cast<const uint32_t*>(src);
                    const uint32_t hi = *reinterpret_cast<const uint32_t*>(src + 2);
                    values[0] = __uint_as_float((lo & 0xffffu) << 16) * inv_scale;
                    values[1] = __uint_as_float(lo & 0xffff0000u) * inv_scale;
                    values[2] = __uint_as_float((hi & 0xffffu) << 16) * inv_scale;
                    values[3] = __uint_as_float(hi & 0xffff0000u) * inv_scale;
                } else {
                    values[0] = i + 0 < intermediate_hidden ?
                        bf16_bits_to_float(src[0]) * inv_scale : 0.0f;
                    values[1] = i + 1 < intermediate_hidden ?
                        bf16_bits_to_float(src[1]) * inv_scale : 0.0f;
                    values[2] = i + 2 < intermediate_hidden ?
                        bf16_bits_to_float(src[2]) * inv_scale : 0.0f;
                    values[3] = i + 3 < intermediate_hidden ?
                        bf16_bits_to_float(src[3]) * inv_scale : 0.0f;
                }
                const uint32_t packed = static_cast<uint32_t>(dcu::pack4_f32_to_fp8(values));
                uint8_t* dst = act_fp8 + row * intermediate_hidden + i;
                if (i + 3 < intermediate_hidden) {
                    *reinterpret_cast<uint32_t*>(dst) = packed;
                } else {
#pragma unroll
                    for (int j = 0; j < 4 && i + j < intermediate_hidden; ++j)
                        dst[j] = static_cast<uint8_t>((packed >> (8 * j)) & 0xffu);
                }
            }
        }
    }
}

template <
    typename TileConfig,
    bool kUseChunkActScale = false,
    int kActScaleChunkBytes = TileConfig::kKStageBytes>
__device__ static inline void compute_route_mmac_mtile16_l2_chunk(
    const int local_expert,
    const int hidden_start,
    const int valid_rows,
    const int64_t tile_meta_base,
    const int hidden,
    const int intermediate_hidden,
    uint16_t* const* tile_combine_row_ptrs,
    const uint8_t* l2_weights, const float* l2_weights_sf,
    const uint8_t* act_fp8,
    const float* act_scale,
    const float* act_chunk_scale,
    const int num_inter_chunks,
    uint4* lds_a_stage) {
    constexpr int kTileM = TileConfig::kTileM;
    static_assert(TileConfig::kNChunksPerWave == 8 || TileConfig::kNChunksPerWave == 4);
    static_assert(TileConfig::kDoubleBuffer);
    static_assert(!kUseChunkActScale || kActScaleChunkBytes > 0);
    static_assert(!kUseChunkActScale || kActScaleChunkBytes % 32 == 0);
    static_assert(!kUseChunkActScale || TileConfig::kKStageBytes % kActScaleChunkBytes == 0);
    const int wave_id = static_cast<int>(threadIdx.x >> 6);
    const int lane = static_cast<int>(threadIdx.x & 63);
    const int lane_m = lane & 15;
    const int lane_n = lane & 15;
    const int lane_k = (lane >> 4) * 8;
    const int lane_group = lane >> 4;
    const int c_n_base = lane_group;
    const bool row_valid = lane_m < valid_rows;
    constexpr int kLdsLoaderWaves = 2;
    constexpr int kWaveSize = 64;
    constexpr int kLdsLoaderThreads = kLdsLoaderWaves * kWaveSize;
    const bool lds_loader_wave = wave_id < kLdsLoaderWaves;
    const int l2_rows = hidden;
    const float* l2_weights_sf_expert =
        l2_weights_sf + static_cast<int64_t>(local_expert) * l2_rows;
#if defined(__gfx938__)
    const uint32x4_t l2_weight_resource = make_dcu_buffer_resource(l2_weights);
#endif
    const int h_base0 =
        hidden_start + wave_id * TileConfig::kNChunksPerWave * TileConfig::kMmacTileN;
    const int h_base1 = h_base0 + TileConfig::kMmacTileN;
    const int h_base2 = h_base0 + 2 * TileConfig::kMmacTileN;
    const int h_base3 = h_base0 + 3 * TileConfig::kMmacTileN;
    const int h_base4 = h_base0 + 4 * TileConfig::kMmacTileN;
    const int h_base5 = h_base0 + 5 * TileConfig::kMmacTileN;
    const int h_base6 = h_base0 + 6 * TileConfig::kMmacTileN;
    const int h_base7 = h_base0 + 7 * TileConfig::kMmacTileN;
    const int h_row0 = h_base0 + lane_n;
    const int h_row1 = h_base1 + lane_n;
    const int h_row2 = h_base2 + lane_n;
    const int h_row3 = h_base3 + lane_n;
    const int h_row4 = h_base4 + lane_n;
    const int h_row5 = h_base5 + lane_n;
    const int h_row6 = h_base6 + lane_n;
    const int h_row7 = h_base7 + lane_n;
    const bool hidden_valid0 = h_row0 < hidden;
    const bool hidden_valid1 = h_row1 < hidden;
    const bool hidden_valid2 = h_row2 < hidden;
    const bool hidden_valid3 = h_row3 < hidden;
    const bool hidden_valid4 = h_row4 < hidden;
    const bool hidden_valid5 = h_row5 < hidden;
    const bool hidden_valid6 = h_row6 < hidden;
    const bool hidden_valid7 = h_row7 < hidden;
    const int64_t h_base_offset0 = marlin_nt_kpack2_row_base_offset(
        local_expert, h_row0, l2_rows, intermediate_hidden);
    const int64_t h_base_offset1 = marlin_nt_kpack2_row_base_offset(
        local_expert, h_row1, l2_rows, intermediate_hidden);
    const int64_t h_base_offset2 = marlin_nt_kpack2_row_base_offset(
        local_expert, h_row2, l2_rows, intermediate_hidden);
    const int64_t h_base_offset3 = marlin_nt_kpack2_row_base_offset(
        local_expert, h_row3, l2_rows, intermediate_hidden);
    const int64_t h_base_offset4 = marlin_nt_kpack2_row_base_offset(
        local_expert, h_row4, l2_rows, intermediate_hidden);
    const int64_t h_base_offset5 = marlin_nt_kpack2_row_base_offset(
        local_expert, h_row5, l2_rows, intermediate_hidden);
    const int64_t h_base_offset6 = marlin_nt_kpack2_row_base_offset(
        local_expert, h_row6, l2_rows, intermediate_hidden);
    const int64_t h_base_offset7 = marlin_nt_kpack2_row_base_offset(
        local_expert, h_row7, l2_rows, intermediate_hidden);

    uint16_t* combine_row = nullptr;
    if (row_valid) {
        combine_row = tile_combine_row_ptrs[tile_meta_base + lane_m];
    }

    dcu::int32x2_t zero8;
    zero8.x = 0;
    zero8.y = 0;

    dcu::float32x4_t l2_acc0{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_acc1{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_acc2{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_acc3{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_acc4{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_acc5{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_acc6{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_acc7{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_sum0{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_sum1{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_sum2{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_sum3{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_sum4{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_sum5{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_sum6{0.0f, 0.0f, 0.0f, 0.0f};
    dcu::float32x4_t l2_sum7{0.0f, 0.0f, 0.0f, 0.0f};
    constexpr bool kUseLdsBForMmac = true;
    stage_mtile16_a_operand_to_lds_nowait<
        TileConfig::kKStageBytes,
        TileConfig::kLdsAStageStrideBytes>(
        act_fp8, valid_rows, 0, intermediate_hidden, intermediate_hidden, lds_a_stage);
    if constexpr (TileConfig::kL2LdsBStageRows > 0) {
        stage_l2_b_operand_to_lds_nowait_coop<TileConfig>(
            l2_weights,
            local_expert,
            hidden_start,
            0,
            hidden,
            intermediate_hidden,
            lds_a_stage + TileConfig::kLdsAStageVecs,
            threadIdx.x,
            blockDim.x);
    }
    wait_mtile16_a_operand_to_lds();
    __syncthreads();
    for (int k_stage = 0, lds_buffer_idx = 0;
         k_stage < intermediate_hidden;
         k_stage += TileConfig::kKStageBytes, lds_buffer_idx ^= 1) {
        if constexpr (kUseChunkActScale) {
            dcu_vec4_zero(l2_acc0);
            dcu_vec4_zero(l2_acc1);
            dcu_vec4_zero(l2_acc2);
            dcu_vec4_zero(l2_acc3);
            dcu_vec4_zero(l2_acc4);
            dcu_vec4_zero(l2_acc5);
            dcu_vec4_zero(l2_acc6);
            dcu_vec4_zero(l2_acc7);
        }
        const int next_k_stage = k_stage + TileConfig::kKStageBytes;
        const int next_lds_buffer_idx = lds_buffer_idx ^ 1;
        if (lds_loader_wave && next_k_stage < intermediate_hidden) {
            stage_mtile16_a_operand_to_lds_nowait_coop<
                TileConfig::kKStageBytes,
                TileConfig::kLdsAStageStrideBytes>(
                act_fp8,
                valid_rows,
                next_k_stage,
                intermediate_hidden,
                intermediate_hidden,
                lds_a_stage + next_lds_buffer_idx * TileConfig::kLdsStageVecs,
                threadIdx.x,
                kLdsLoaderThreads);
            if constexpr (TileConfig::kL2LdsBStageRows > 0 && kUseLdsBForMmac) {
                uint4* next_stage = lds_a_stage + next_lds_buffer_idx * TileConfig::kLdsStageVecs;
                stage_l2_b_operand_to_lds_nowait_coop<TileConfig>(
                    l2_weights,
                    local_expert,
                    hidden_start,
                    next_k_stage,
                    hidden,
                    intermediate_hidden,
                    next_stage + TileConfig::kLdsAStageVecs,
                    threadIdx.x,
                    kLdsLoaderThreads);
            }
        }
        const auto* current_stage = lds_a_stage + lds_buffer_idx * TileConfig::kLdsStageVecs;
        const auto* lds_a = reinterpret_cast<const uint8_t*>(current_stage);
        const auto* lds_b = reinterpret_cast<const uint8_t*>(
            current_stage + TileConfig::kLdsAStageVecs);
        const int k_stage_end = min(k_stage + TileConfig::kKStageBytes, intermediate_hidden);
#pragma nounroll
        for (int k_base = k_stage; k_base < k_stage_end; k_base += 32) {
            const int k_idx = k_base + lane_k;
            const int lds_k_idx = k_idx - k_stage;
            const int a_byte_offset =
                lane_m * TileConfig::kLdsAStageStrideBytes + lds_k_idx;
            dcu::int32x2_t a_vec = zero8;
            if (row_valid)
                a_vec = load_fp8_lds_b64_nowait(lds_a, a_byte_offset);
            if constexpr (TileConfig::kL2LdsBStageRows > 0 && kUseLdsBForMmac) {
                const int b_base_row =
                    wave_id * TileConfig::kNChunksPerWave * TileConfig::kMmacTileN + lane_n;
                const int b_byte0 = (b_base_row + 0) * TileConfig::kKStageBytes + lds_k_idx;
                const int b_byte1 =
                    (b_base_row + TileConfig::kMmacTileN) *
                    TileConfig::kKStageBytes + lds_k_idx;
                const int b_byte2 =
                    (b_base_row + 2 * TileConfig::kMmacTileN) *
                    TileConfig::kKStageBytes + lds_k_idx;
                const int b_byte3 =
                    (b_base_row + 3 * TileConfig::kMmacTileN) *
                    TileConfig::kKStageBytes + lds_k_idx;
                int32x4_t b_pair0 = load_fp8_lds_b128_nowait(lds_b, b_byte0);
                int32x4_t b_pair1{0, 0, 0, 0};
                int32x4_t b_pair2{0, 0, 0, 0};
                int32x4_t b_pair3{0, 0, 0, 0};
                if constexpr (TileConfig::kNChunksPerWave >= 2)
                    b_pair1 = load_fp8_lds_b128_nowait(lds_b, b_byte1);
                if constexpr (TileConfig::kNChunksPerWave >= 3)
                    b_pair2 = load_fp8_lds_b128_nowait(lds_b, b_byte2);
                if constexpr (TileConfig::kNChunksPerWave >= 4)
                    b_pair3 = load_fp8_lds_b128_nowait(lds_b, b_byte3);
                wait_lds_reads();
                const dcu::int32x2_t w_vec0 = select_fp8_pair_from_b128(b_pair0, b_byte0);
                l2_acc0 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec0, l2_acc0);
                if constexpr (TileConfig::kNChunksPerWave >= 2) {
                    const dcu::int32x2_t w_vec1 = select_fp8_pair_from_b128(b_pair1, b_byte1);
                    l2_acc1 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec1, l2_acc1);
                }
                if constexpr (TileConfig::kNChunksPerWave >= 3) {
                    const dcu::int32x2_t w_vec2 = select_fp8_pair_from_b128(b_pair2, b_byte2);
                    l2_acc2 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec2, l2_acc2);
                }
                if constexpr (TileConfig::kNChunksPerWave >= 4) {
                    const dcu::int32x2_t w_vec3 = select_fp8_pair_from_b128(b_pair3, b_byte3);
                    l2_acc3 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec3, l2_acc3);
                }
            } else {
            wait_lds_reads();
#if defined(__gfx938__)
            const int weight_k_offset = marlin_nt_kpack2_k_offset(k_idx);
            if constexpr (TileConfig::kNChunksPerWave <= 4) {
                const dcu::int32x2_t w_vec0 = hidden_valid0
                    ? dcu::pack8_fp8(l2_weights + h_base_offset0 + weight_k_offset)
                    : zero8;
                const dcu::int32x2_t w_vec1 = hidden_valid1
                    ? dcu::pack8_fp8(l2_weights + h_base_offset1 + weight_k_offset)
                    : zero8;
                const dcu::int32x2_t w_vec2 = hidden_valid2
                    ? dcu::pack8_fp8(l2_weights + h_base_offset2 + weight_k_offset)
                    : zero8;
                l2_acc0 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec0, l2_acc0);
                if constexpr (TileConfig::kNChunksPerWave >= 2)
                    l2_acc1 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec1, l2_acc1);
                if constexpr (TileConfig::kNChunksPerWave >= 3)
                    l2_acc2 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec2, l2_acc2);
                if constexpr (TileConfig::kNChunksPerWave >= 4) {
                    const dcu::int32x2_t w_vec3 = hidden_valid3
                        ? dcu::pack8_fp8(l2_weights + h_base_offset3 + weight_k_offset)
                        : zero8;
                    l2_acc3 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec3, l2_acc3);
                }
            } else {
                fp8_mmac8_buffer_asm(
                    l2_acc0, l2_acc1, l2_acc2, l2_acc3,
                    l2_acc4, l2_acc5, l2_acc6, l2_acc7,
                    a_vec,
                    l2_weight_resource,
                    static_cast<uint32_t>(h_base_offset0 + weight_k_offset),
                    static_cast<uint32_t>(h_base_offset1 + weight_k_offset),
                    static_cast<uint32_t>(h_base_offset2 + weight_k_offset),
                    static_cast<uint32_t>(h_base_offset3 + weight_k_offset),
                    static_cast<uint32_t>(h_base_offset4 + weight_k_offset),
                    static_cast<uint32_t>(h_base_offset5 + weight_k_offset),
                    static_cast<uint32_t>(h_base_offset6 + weight_k_offset),
                    static_cast<uint32_t>(h_base_offset7 + weight_k_offset));
            }
#else
            const dcu::int32x2_t w_vec0 = hidden_valid0
                ? pack8_fp8_weight_marlin_row_base(l2_weights, h_base_offset0, k_idx)
                : zero8;
            const dcu::int32x2_t w_vec1 = hidden_valid1
                ? pack8_fp8_weight_marlin_row_base(l2_weights, h_base_offset1, k_idx)
                : zero8;
            const dcu::int32x2_t w_vec2 = hidden_valid2
                ? pack8_fp8_weight_marlin_row_base(l2_weights, h_base_offset2, k_idx)
                : zero8;
            const dcu::int32x2_t w_vec3 = hidden_valid3
                ? pack8_fp8_weight_marlin_row_base(l2_weights, h_base_offset3, k_idx)
                : zero8;
            const dcu::int32x2_t w_vec4 = hidden_valid4
                ? pack8_fp8_weight_marlin_row_base(l2_weights, h_base_offset4, k_idx)
                : zero8;
            const dcu::int32x2_t w_vec5 = hidden_valid5
                ? pack8_fp8_weight_marlin_row_base(l2_weights, h_base_offset5, k_idx)
                : zero8;
            const dcu::int32x2_t w_vec6 = hidden_valid6
                ? pack8_fp8_weight_marlin_row_base(l2_weights, h_base_offset6, k_idx)
                : zero8;
            const dcu::int32x2_t w_vec7 = hidden_valid7
                ? pack8_fp8_weight_marlin_row_base(l2_weights, h_base_offset7, k_idx)
                : zero8;
            l2_acc0 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec0, l2_acc0);
            l2_acc1 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec1, l2_acc1);
            l2_acc2 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec2, l2_acc2);
            l2_acc3 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec3, l2_acc3);
            l2_acc4 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec4, l2_acc4);
            l2_acc5 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec5, l2_acc5);
            l2_acc6 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec6, l2_acc6);
            l2_acc7 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, w_vec7, l2_acc7);
#endif
            }
            if constexpr (kUseChunkActScale) {
                const int next_k_base = k_base + 32;
                if (next_k_base == k_stage_end ||
                    (next_k_base % kActScaleChunkBytes) == 0) {
                    const int act_chunk_id = k_base / kActScaleChunkBytes;
                    const float row_chunk_scale =
                        row_valid ? act_chunk_scale[lane_m * num_inter_chunks + act_chunk_id] : 1.0f;
                    dcu_vec4_accum_scaled(l2_sum0, l2_acc0, row_chunk_scale);
                    dcu_vec4_accum_scaled(l2_sum1, l2_acc1, row_chunk_scale);
                    dcu_vec4_accum_scaled(l2_sum2, l2_acc2, row_chunk_scale);
                    dcu_vec4_accum_scaled(l2_sum3, l2_acc3, row_chunk_scale);
                    dcu_vec4_accum_scaled(l2_sum4, l2_acc4, row_chunk_scale);
                    dcu_vec4_accum_scaled(l2_sum5, l2_acc5, row_chunk_scale);
                    dcu_vec4_accum_scaled(l2_sum6, l2_acc6, row_chunk_scale);
                    dcu_vec4_accum_scaled(l2_sum7, l2_acc7, row_chunk_scale);
                    if (next_k_base < k_stage_end) {
                        dcu_vec4_zero(l2_acc0);
                        dcu_vec4_zero(l2_acc1);
                        dcu_vec4_zero(l2_acc2);
                        dcu_vec4_zero(l2_acc3);
                        dcu_vec4_zero(l2_acc4);
                        dcu_vec4_zero(l2_acc5);
                        dcu_vec4_zero(l2_acc6);
                        dcu_vec4_zero(l2_acc7);
                    }
                }
            }
        }
        if (next_k_stage < intermediate_hidden)
            wait_mtile16_a_operand_to_lds();
        __syncthreads();
    }

    if (row_valid) {
        if constexpr (kUseChunkActScale) {
            l2_acc0 = l2_sum0;
            l2_acc1 = l2_sum1;
            l2_acc2 = l2_sum2;
            l2_acc3 = l2_sum3;
            l2_acc4 = l2_sum4;
            l2_acc5 = l2_sum5;
            l2_acc6 = l2_sum6;
            l2_acc7 = l2_sum7;
        }
        const float row_act_scale = kUseChunkActScale ? 1.0f : act_scale[lane_m];
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int n = c_n_base + 4 * slot;
            const int h_idx = h_base0 + n;
            if (h_idx < hidden) {
                const float w2_scale = l2_weights_sf_expert[h_idx];
                combine_row[h_idx] = float_to_bf16_bits(
                    vec4_get(l2_acc0, slot) * row_act_scale * w2_scale);
            }
        }
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            const int n = c_n_base + 4 * slot;
            const int h_idx = h_base1 + n;
            if (h_idx < hidden) {
                const float w2_scale = l2_weights_sf_expert[h_idx];
                combine_row[h_idx] = float_to_bf16_bits(
                    vec4_get(l2_acc1, slot) * row_act_scale * w2_scale);
            }
        }
        if constexpr (TileConfig::kNChunksPerWave >= 3) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int n = c_n_base + 4 * slot;
                const int h_idx = h_base2 + n;
                if (h_idx < hidden) {
                    const float w2_scale = l2_weights_sf_expert[h_idx];
                    combine_row[h_idx] = float_to_bf16_bits(
                        vec4_get(l2_acc2, slot) * row_act_scale * w2_scale);
                }
            }
        }
        if constexpr (TileConfig::kNChunksPerWave >= 4) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int n = c_n_base + 4 * slot;
                const int h_idx = h_base3 + n;
                if (h_idx < hidden) {
                    const float w2_scale = l2_weights_sf_expert[h_idx];
                    combine_row[h_idx] = float_to_bf16_bits(
                        vec4_get(l2_acc3, slot) * row_act_scale * w2_scale);
                }
            }
        }
        if constexpr (TileConfig::kNChunksPerWave == 8) {
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int n = c_n_base + 4 * slot;
                const int h_idx = h_base4 + n;
                if (h_idx < hidden) {
                    const float w2_scale = l2_weights_sf_expert[h_idx];
                    combine_row[h_idx] = float_to_bf16_bits(
                        vec4_get(l2_acc4, slot) * row_act_scale * w2_scale);
                }
            }
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int n = c_n_base + 4 * slot;
                const int h_idx = h_base5 + n;
                if (h_idx < hidden) {
                    const float w2_scale = l2_weights_sf_expert[h_idx];
                    combine_row[h_idx] = float_to_bf16_bits(
                        vec4_get(l2_acc5, slot) * row_act_scale * w2_scale);
                }
            }
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int n = c_n_base + 4 * slot;
                const int h_idx = h_base6 + n;
                if (h_idx < hidden) {
                    const float w2_scale = l2_weights_sf_expert[h_idx];
                    combine_row[h_idx] = float_to_bf16_bits(
                        vec4_get(l2_acc6, slot) * row_act_scale * w2_scale);
                }
            }
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int n = c_n_base + 4 * slot;
                const int h_idx = h_base7 + n;
                if (h_idx < hidden) {
                    const float w2_scale = l2_weights_sf_expert[h_idx];
                    combine_row[h_idx] = float_to_bf16_bits(
                        vec4_get(l2_acc7, slot) * row_act_scale * w2_scale);
                }
            }
        }
    }
}

} // namespace deep_gemm::mega
