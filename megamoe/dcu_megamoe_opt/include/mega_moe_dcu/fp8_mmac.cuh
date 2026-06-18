#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>

namespace deep_gemm::mega::dcu {

using int32x2_t = int __attribute__((ext_vector_type(2)));
using float32x4_t = float __attribute__((ext_vector_type(4)));

struct Fp8Mmac16x16x32 {
    static constexpr int kWaveSize = 64;
    static constexpr int kM = 16;
    static constexpr int kN = 16;
    static constexpr int kK = 32;
    static constexpr int kPackedKPerLane = 8;
    static constexpr int kCValuesPerLane = 4;

    __device__ static inline int lane_id() {
        return threadIdx.x & (kWaveSize - 1);
    }

    __device__ static inline int a_m(const int lane) {
        return lane & 15;
    }

    __device__ static inline int a_k_offset(const int lane) {
        return (lane >> 4) * kPackedKPerLane;
    }

    __device__ static inline int b_n(const int lane) {
        return lane & 15;
    }

    __device__ static inline int b_k_offset(const int lane) {
        return (lane >> 4) * kPackedKPerLane;
    }

    __device__ static inline int c_m(const int lane) {
        return lane & 15;
    }

    __device__ static inline int c_n(const int lane, const int slot) {
        return (lane >> 4) + 4 * slot;
    }
};

__device__ static inline int pack4_bytes(const uint8_t* ptr) {
    return *reinterpret_cast<const int*>(ptr);
}

__device__ static inline int32x2_t pack8_fp8(const uint8_t* ptr) {
    return *reinterpret_cast<const int32x2_t*>(ptr);
}

__device__ static inline uint8_t load_ue8m0_byte(const uint8_t* sf,
                                                 const int row_idx,
                                                 const int k_group,
                                                 const int sf_groups_per_row) {
    return sf[static_cast<int64_t>(row_idx) * sf_groups_per_row + k_group];
}

__device__ static inline uint8_t ue8m0_exp_from_amax(const float amax) {
    const float sf = fmaxf(amax / 448.0f, 1.0e-4f);
    const uint32_t bits = __float_as_uint(sf);
    int exp = static_cast<int>((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0);
    exp = exp < 1 ? 1 : exp;
    exp = exp > 254 ? 254 : exp;
    return static_cast<uint8_t>(exp);
}

__device__ static inline float cvt_f32_fp8(const uint8_t value) {
#if defined(__gfx938__)
    return __builtin_hcu_cvt_f32_fp8(value, false, 0, 0);
#else
    (void)value;
    assert(false && "cvt_f32_fp8 is only supported on gfx938");
    return 0.0f;
#endif
}

template <bool kHighHalf = false>
__device__ static inline int cvt_pk_fp8_f32(const float a,
                                            const float b,
                                            const int old_value) {
#if defined(__gfx938__)
    return __builtin_hcu_cvt_pk_fp8_f32(a, b, old_value, kHighHalf);
#else
    (void)a;
    (void)b;
    assert(false && "cvt_pk_fp8_f32 is only supported on gfx938");
    return old_value;
#endif
}

__device__ static inline int pack4_f32_to_fp8(const float* values) {
    int out = 0;
    out = cvt_pk_fp8_f32<false>(values[0], values[1], out);
    out = cvt_pk_fp8_f32<true>(values[2], values[3], out);
    return out;
}

__device__ static inline int32x2_t pack8_f32_to_fp8(const float* values) {
    int32x2_t out;
    out.x = pack4_f32_to_fp8(values);
    out.y = pack4_f32_to_fp8(values + 4);
    return out;
}

template <bool kLit = false, bool kLts = false>
__device__ static inline float32x4_t mmac_f32_16x16x32_fp8_fp8(
    const int32x2_t a,
    const int32x2_t b,
    const float32x4_t c) {
#if defined(__gfx938__)
    return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a, b, c, kLit, kLts);
#else
    (void)a;
    (void)b;
    assert(false && "mmac_f32_16x16x32_fp8_fp8 is only supported on gfx938");
    return c;
#endif
}

} // namespace deep_gemm::mega::dcu
