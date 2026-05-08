#pragma once

#include <deep_gemm/common/mega_moe_dcu.cuh>

namespace deep_gemm::mega {

static constexpr int kMTile16L1NChunksPerWave = 4;
static constexpr int kMTile16L2NChunksPerWave = 8;

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
    uint16_t* act_bf16) {
    constexpr int kTileM = 16;
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
    const int inter_base0 = inter_start + wave_id * kMTile16L1NChunksPerWave * 16;
    const int inter_base1 = inter_base0 + 16;
    const int inter_base2 = inter_base0 + 32;
    const int inter_base3 = inter_base0 + 48;
    const bool inter_valid0 = inter_base0 + lane_n < intermediate_hidden;
    const bool inter_valid1 = inter_base1 + lane_n < intermediate_hidden;
    const bool inter_valid2 = inter_base2 + lane_n < intermediate_hidden;
    const bool inter_valid3 = inter_base3 + lane_n < intermediate_hidden;

    const int64_t meta_idx = tile_meta_base + lane_m;
    const uint8_t* x_row = nullptr;
    float route_weight = 0.0f;
    float x_scale = 0.0f;
    if (row_valid) {
        x_row = tile_x_pool + meta_idx * hidden;
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
    for (int k_base = 0; k_base < hidden; k_base += 32) {
        const int k_idx = k_base + lane_k;
        const dcu::int32x2_t a_vec = row_valid
            ? dcu::pack8_fp8(x_row + k_idx)
            : zero8;
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
        gate_acc0 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, gate_vec0, gate_acc0);
        up_acc0 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, up_vec0, up_acc0);
        gate_acc1 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, gate_vec1, gate_acc1);
        up_acc1 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, up_vec1, up_acc1);
        gate_acc2 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, gate_vec2, gate_acc2);
        up_acc2 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, up_vec2, up_acc2);
        gate_acc3 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, gate_vec3, gate_acc3);
        up_acc3 = dcu::mmac_f32_16x16x32_fp8_fp8(a_vec, up_vec3, up_acc3);
    }

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
                if (isfinite(activation_clamp)) {
                    gate = fminf(fmaxf(gate, -activation_clamp), activation_clamp);
                    up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                }
                const float silu = fast_silu(gate);
                act_bf16[lane_m * intermediate_hidden + inter_idx] =
                    float_to_bf16_bits(silu * up * route_weight);
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
                if (isfinite(activation_clamp)) {
                    gate = fminf(fmaxf(gate, -activation_clamp), activation_clamp);
                    up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                }
                const float silu = fast_silu(gate);
                act_bf16[lane_m * intermediate_hidden + inter_idx] =
                    float_to_bf16_bits(silu * up * route_weight);
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
                if (isfinite(activation_clamp)) {
                    gate = fminf(fmaxf(gate, -activation_clamp), activation_clamp);
                    up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                }
                const float silu = fast_silu(gate);
                act_bf16[lane_m * intermediate_hidden + inter_idx] =
                    float_to_bf16_bits(silu * up * route_weight);
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
                if (isfinite(activation_clamp)) {
                    gate = fminf(fmaxf(gate, -activation_clamp), activation_clamp);
                    up = fminf(fmaxf(up, -activation_clamp), activation_clamp);
                }
                const float silu = fast_silu(gate);
                act_bf16[lane_m * intermediate_hidden + inter_idx] =
                    float_to_bf16_bits(silu * up * route_weight);
            }
        }
    }
}

__device__ static inline void quant_bf16_act_channelwise_mtile16_global(
    const uint16_t* act_bf16,
    uint8_t* act_fp8,
    float* act_scale,
    const int valid_rows,
    const int intermediate_hidden) {
    const int wave_id = static_cast<int>(threadIdx.x >> 6);
    const int wave_count = static_cast<int>(blockDim.x >> 6);
    const int lane = static_cast<int>(threadIdx.x & 63);

    for (int row = wave_id; row < 16; row += wave_count) {
        const bool row_valid = row < valid_rows;
        float local_amax = 0.0f;
        if (row_valid) {
            for (int i = lane * 4; i < intermediate_hidden; i += 64 * 4) {
                const uint16_t* src = act_bf16 + row * intermediate_hidden + i;
                if (i + 3 < intermediate_hidden) {
                    const uint32_t lo = *reinterpret_cast<const uint32_t*>(src);
                    const uint32_t hi = *reinterpret_cast<const uint32_t*>(src + 2);
                    local_amax = fmaxf(local_amax, fabsf(__uint_as_float((lo & 0xffffu) << 16)));
                    local_amax = fmaxf(local_amax, fabsf(__uint_as_float(lo & 0xffff0000u)));
                    local_amax = fmaxf(local_amax, fabsf(__uint_as_float((hi & 0xffffu) << 16)));
                    local_amax = fmaxf(local_amax, fabsf(__uint_as_float(hi & 0xffff0000u)));
                } else {
#pragma unroll
                    for (int j = 0; j < 4 && i + j < intermediate_hidden; ++j) {
                        local_amax = fmaxf(local_amax, fabsf(bf16_bits_to_float(src[j])));
                    }
                }
            }
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
    const float* act_scale) {
    constexpr int kTileM = 16;
    const int wave_id = static_cast<int>(threadIdx.x >> 6);
    const int lane = static_cast<int>(threadIdx.x & 63);
    const int lane_m = lane & 15;
    const int lane_n = lane & 15;
    const int lane_k = (lane >> 4) * 8;
    const int lane_group = lane >> 4;
    const int c_n_base = lane_group;
    const bool row_valid = lane_m < valid_rows;
    const int l2_rows = hidden;
    const float* l2_weights_sf_expert =
        l2_weights_sf + static_cast<int64_t>(local_expert) * l2_rows;
    const int h_base0 = hidden_start + wave_id * kMTile16L2NChunksPerWave * 16;
    const int h_base1 = h_base0 + 16;
    const int h_base2 = h_base0 + 32;
    const int h_base3 = h_base0 + 48;
    const int h_base4 = h_base0 + 64;
    const int h_base5 = h_base0 + 80;
    const int h_base6 = h_base0 + 96;
    const int h_base7 = h_base0 + 112;
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
    for (int k_base = 0; k_base < intermediate_hidden; k_base += 32) {
        const int k_idx = k_base + lane_k;
        const dcu::int32x2_t a_vec =
            row_valid ? dcu::pack8_fp8(act_fp8 + lane_m * intermediate_hidden + k_idx) : zero8;
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
    }

    if (row_valid) {
        const float row_act_scale = act_scale[lane_m];
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

} // namespace deep_gemm::mega
