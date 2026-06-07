#pragma once
//
// vkm — SIMD backend for 128-bit / 4-wide float.
//
// This is the only file that touches platform intrinsics. Everything above it
// works on plain `float*` and lets this layer decide how to crunch four lanes
// at once. We provide three backends, selected at compile time:
//
//   * NEON  — AArch64 (Apple Silicon, ARM servers/phones)
//   * SSE   — x86-64 (SSE2 baseline; horizontal add via shuffles)
//   * scalar— portable fallback so the library always builds
//
// The public surface is intentionally tiny: a `f32x4` register type plus the
// handful of ops the vector code needs. We grow it as higher layers demand.
//
#include <cstddef>

// Backend selection and intrinsic headers — these MUST be included at file
// scope. Pulling <arm_neon.h>/<immintrin.h> inside a namespace would drag every
// transitively-included system typedef into that namespace.
// Define VKM_FORCE_SCALAR to bypass intrinsics (handy for debugging and for
// diffing SIMD vs scalar results in tests).
#if defined(VKM_FORCE_SCALAR)
#define VKM_SIMD_SCALAR 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define VKM_SIMD_NEON 1
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define VKM_SIMD_SSE 1
#include <immintrin.h>
#else
#define VKM_SIMD_SCALAR 1
#endif

namespace vkm::detail {

#if VKM_SIMD_NEON

using f32x4 = float32x4_t;

[[nodiscard]] inline f32x4 load4(const float* p) noexcept { return vld1q_f32(p); }
inline void store4(float* p, f32x4 v) noexcept { vst1q_f32(p, v); }
[[nodiscard]] inline f32x4 splat(float s) noexcept { return vdupq_n_f32(s); }

[[nodiscard]] inline f32x4 add(f32x4 a, f32x4 b) noexcept { return vaddq_f32(a, b); }
[[nodiscard]] inline f32x4 sub(f32x4 a, f32x4 b) noexcept { return vsubq_f32(a, b); }
[[nodiscard]] inline f32x4 mul(f32x4 a, f32x4 b) noexcept { return vmulq_f32(a, b); }
[[nodiscard]] inline f32x4 div(f32x4 a, f32x4 b) noexcept { return vdivq_f32(a, b); }
[[nodiscard]] inline f32x4 neg(f32x4 a) noexcept { return vnegq_f32(a); }
// Horizontal sum across the four lanes (AArch64 has a single instruction).
[[nodiscard]] inline float hsum(f32x4 v) noexcept { return vaddvq_f32(v); }

#elif VKM_SIMD_SSE

using f32x4 = __m128;

[[nodiscard]] inline f32x4 load4(const float* p) noexcept { return _mm_loadu_ps(p); }
inline void store4(float* p, f32x4 v) noexcept { _mm_storeu_ps(p, v); }
[[nodiscard]] inline f32x4 splat(float s) noexcept { return _mm_set1_ps(s); }

[[nodiscard]] inline f32x4 add(f32x4 a, f32x4 b) noexcept { return _mm_add_ps(a, b); }
[[nodiscard]] inline f32x4 sub(f32x4 a, f32x4 b) noexcept { return _mm_sub_ps(a, b); }
[[nodiscard]] inline f32x4 mul(f32x4 a, f32x4 b) noexcept { return _mm_mul_ps(a, b); }
[[nodiscard]] inline f32x4 div(f32x4 a, f32x4 b) noexcept { return _mm_div_ps(a, b); }
[[nodiscard]] inline f32x4 neg(f32x4 a) noexcept { return _mm_sub_ps(_mm_setzero_ps(), a); }
// Horizontal add, SSE2-only (the backend is gated on __SSE2__, so we must not
// reach for SSE3's haddps here): shuffle+add the two halves down into lane 0.
[[nodiscard]] inline float hsum(f32x4 v) noexcept {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)); // [b,a,d,c]
    __m128 sums = _mm_add_ps(v, shuf);                           // a+b, b+a, c+d, d+c
    shuf = _mm_movehl_ps(shuf, sums);                            // c+d into low lane
    sums = _mm_add_ss(sums, shuf);                               // (a+b)+(c+d)
    return _mm_cvtss_f32(sums);
}

#else // VKM_SIMD_SCALAR

// Portable fallback. Trivially copyable POD so it behaves like a register.
struct f32x4 { float v[4]; };

[[nodiscard]] inline f32x4 load4(const float* p) noexcept { return {{p[0], p[1], p[2], p[3]}}; }
inline void store4(float* p, f32x4 a) noexcept { p[0] = a.v[0]; p[1] = a.v[1]; p[2] = a.v[2]; p[3] = a.v[3]; }
[[nodiscard]] inline f32x4 splat(float s) noexcept { return {{s, s, s, s}}; }

[[nodiscard]] inline f32x4 add(f32x4 a, f32x4 b) noexcept { return {{a.v[0]+b.v[0], a.v[1]+b.v[1], a.v[2]+b.v[2], a.v[3]+b.v[3]}}; }
[[nodiscard]] inline f32x4 sub(f32x4 a, f32x4 b) noexcept { return {{a.v[0]-b.v[0], a.v[1]-b.v[1], a.v[2]-b.v[2], a.v[3]-b.v[3]}}; }
[[nodiscard]] inline f32x4 mul(f32x4 a, f32x4 b) noexcept { return {{a.v[0]*b.v[0], a.v[1]*b.v[1], a.v[2]*b.v[2], a.v[3]*b.v[3]}}; }
[[nodiscard]] inline f32x4 div(f32x4 a, f32x4 b) noexcept { return {{a.v[0]/b.v[0], a.v[1]/b.v[1], a.v[2]/b.v[2], a.v[3]/b.v[3]}}; }
[[nodiscard]] inline f32x4 neg(f32x4 a) noexcept { return {{-a.v[0], -a.v[1], -a.v[2], -a.v[3]}}; }
[[nodiscard]] inline float hsum(f32x4 a) noexcept { return a.v[0] + a.v[1] + a.v[2] + a.v[3]; }

#endif

} // namespace vkm::detail
