#pragma once
//
// vkm — GPU vector packing: normalized-integer quantization + octahedral unit
// vector encoding. Header-only; everything is inline/noexcept, and the parts that
// don't need a sqrt are constexpr.
//
// Ported and generalized from the assetc asset compiler (runtime_mesh.cpp):
// assetc only *encodes* (snorm16 + octahedral normals/tangents); this header adds
// the matching decoders and the 8/10-bit + unorm variants.
//
// Octahedral mapping follows Meyer et al. 2010 ("On Floating-Point Normal
// Vectors"); the decode uses the Cigolle et al. survey form. See oct_project /
// oct_unproject and the Slang snippet below.
//
#include "vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace vkm {

namespace detail {
// signNotZero: +1 for v >= 0 (including +0), -1 for v < 0. (assetc OctSign.)
[[nodiscard]] inline constexpr float oct_sign(float v) noexcept { return v >= 0.0f ? 1.0f : -1.0f; }
// Constexpr |v| and round-half-away-from-zero, so the encoders stay constexpr
// without depending on a constexpr <cmath>. round_half_away matches the
// std::lround the reference uses for every input that isn't an exact half.
[[nodiscard]] inline constexpr float oct_abs(float v) noexcept { return v < 0.0f ? -v : v; }
[[nodiscard]] inline constexpr int round_half_away(float v) noexcept {
    return static_cast<int>(v + (v >= 0.0f ? 0.5f : -0.5f));
}
} // namespace detail

// Packed 2x16-bit pair: 4 bytes, trivially copyable -> memcpy straight into a
// vertex buffer (VK_FORMAT_R16G16_SNORM / _SINT).
struct short2 {
    std::int16_t x, y;
};
static_assert(sizeof(short2) == 4 && alignof(short2) == 2);
static_assert(std::is_trivially_copyable_v<short2>);

// ---- normalized-integer quantization ------------------------------------------
// snorm: float in [-1, 1] <-> signed int, scale 2^(n-1)-1. Encode clamps + rounds;
// decode clamps to [-1, 1] (the most-negative code, e.g. -32768, maps below -1).
// unorm: float in [0, 1] <-> unsigned int, scale 2^n-1.

[[nodiscard]] inline constexpr std::int16_t quantize_snorm16(float v) noexcept {
    return static_cast<std::int16_t>(detail::round_half_away(std::clamp(v, -1.0f, 1.0f) * 32767.0f));
}
[[nodiscard]] inline constexpr float dequantize_snorm16(std::int16_t v) noexcept {
    return std::clamp(static_cast<float>(v) / 32767.0f, -1.0f, 1.0f);
}

[[nodiscard]] inline constexpr std::int8_t quantize_snorm8(float v) noexcept {
    return static_cast<std::int8_t>(detail::round_half_away(std::clamp(v, -1.0f, 1.0f) * 127.0f));
}
[[nodiscard]] inline constexpr float dequantize_snorm8(std::int8_t v) noexcept {
    return std::clamp(static_cast<float>(v) / 127.0f, -1.0f, 1.0f);
}

// 10-bit snorm: value lives in the low 10 bits of an int16 (range [-511, 511]).
[[nodiscard]] inline constexpr std::int16_t quantize_snorm10(float v) noexcept {
    return static_cast<std::int16_t>(detail::round_half_away(std::clamp(v, -1.0f, 1.0f) * 511.0f));
}
[[nodiscard]] inline constexpr float dequantize_snorm10(std::int16_t v) noexcept {
    return std::clamp(static_cast<float>(v) / 511.0f, -1.0f, 1.0f);
}

[[nodiscard]] inline constexpr std::uint16_t quantize_unorm16(float v) noexcept {
    return static_cast<std::uint16_t>(detail::round_half_away(std::clamp(v, 0.0f, 1.0f) * 65535.0f));
}
[[nodiscard]] inline constexpr float dequantize_unorm16(std::uint16_t v) noexcept {
    return static_cast<float>(v) / 65535.0f;
}

[[nodiscard]] inline constexpr std::uint8_t quantize_unorm8(float v) noexcept {
    return static_cast<std::uint8_t>(detail::round_half_away(std::clamp(v, 0.0f, 1.0f) * 255.0f));
}
[[nodiscard]] inline constexpr float dequantize_unorm8(std::uint8_t v) noexcept {
    return static_cast<float>(v) / 255.0f;
}

// 10-bit unorm: value in the low 10 bits of a uint16 (range [0, 1023]).
[[nodiscard]] inline constexpr std::uint16_t quantize_unorm10(float v) noexcept {
    return static_cast<std::uint16_t>(detail::round_half_away(std::clamp(v, 0.0f, 1.0f) * 1023.0f));
}
[[nodiscard]] inline constexpr float dequantize_unorm10(std::uint16_t v) noexcept {
    return static_cast<float>(v) / 1023.0f;
}

// ---- octahedral unit-vector encoding ------------------------------------------
//
// Matching Slang/HLSL decoder (kept in sync with oct_unproject below):
//
//     float3 octDecode(float2 e) {
//         float3 v = float3(e, 1.0 - abs(e.x) - abs(e.y));
//         float  t = max(-v.z, 0.0);
//         v.x += (v.x >= 0.0) ? -t : t;
//         v.y += (v.y >= 0.0) ? -t : t;
//         return normalize(v);
//     }
//
// Vertex formats: normal -> R16G16_SNORM (HW divides by 32767 on fetch); tangent
// -> R16G16_SINT (shader needs the raw int to read the handedness LSB, divides by
// 32767 itself). Decode of either: octDecode(...). Bitangent in the shader:
//     float3 bitangent = cross(normal, tangent) * handedness;

// Fold a unit vector onto the octahedron; result in [-1, 1]^2.
[[nodiscard]] inline constexpr float2 oct_project(float3 n) noexcept {
    const float inv_l1 = 1.0f / (detail::oct_abs(n.x) + detail::oct_abs(n.y) + detail::oct_abs(n.z));
    float2 p{n.x * inv_l1, n.y * inv_l1};
    if (n.z < 0.0f)
        p = float2{(1.0f - detail::oct_abs(p.y)) * detail::oct_sign(p.x),
                   (1.0f - detail::oct_abs(p.x)) * detail::oct_sign(p.y)};
    return p;
}

// Inverse of oct_project: unfold back to a (re-normalized) unit vector.
[[nodiscard]] inline float3 oct_unproject(float2 e) noexcept {
    float3 v{e.x, e.y, 1.0f - detail::oct_abs(e.x) - detail::oct_abs(e.y)};
    const float t = std::max(-v.z, 0.0f);
    v.x += (v.x >= 0.0f) ? -t : t;
    v.y += (v.y >= 0.0f) ? -t : t;
    return normalize(v);
}

// Unit vector -> packed snorm16 octahedral pair (and back).
[[nodiscard]] inline constexpr short2 oct_encode16(float3 n) noexcept {
    const float2 e = oct_project(n);
    return short2{quantize_snorm16(e.x), quantize_snorm16(e.y)};
}
[[nodiscard]] inline float3 oct_decode16(short2 e) noexcept {
    return oct_unproject(float2{dequantize_snorm16(e.x), dequantize_snorm16(e.y)});
}

// ---- tangent packing ----------------------------------------------------------
// Bitangent sign rides in the LSB of .x: 0 => +1, 1 => -1. The 1-LSB nudge is
// ~1/32767 ~= 3e-5 of perturbation on the tangent direction.

[[nodiscard]] inline constexpr short2 oct_encode_tangent(float3 t, float handedness) noexcept {
    short2 e = oct_encode16(t);
    e.x = static_cast<std::int16_t>(e.x & ~std::int16_t{1}); // clear sign LSB
    if (handedness < 0.0f)
        e.x = static_cast<std::int16_t>(e.x | std::int16_t{1});
    return e;
}

// Reads the handedness from packed.x's LSB, then decodes the tangent from packed
// AS-IS (LSB not cleared) so this matches the shader path bit-for-bit. Caller
// rebuilds the bitangent: cross(normal, tangent) * out_handedness.
[[nodiscard]] inline float3 oct_decode_tangent(short2 packed, float& out_handedness) noexcept {
    out_handedness = (packed.x & 1) ? -1.0f : 1.0f;
    return oct_decode16(packed);
}

// ---- compile-time sanity ------------------------------------------------------
static_assert(quantize_snorm16(1.0f) == 32767 && quantize_snorm16(-1.0f) == -32767);
static_assert(quantize_snorm16(2.0f) == 32767);   // clamps on encode
static_assert(dequantize_snorm16(-32768) == -1.0f); // clamps on decode
static_assert(quantize_unorm8(1.0f) == 255 && quantize_unorm8(0.0f) == 0);
static_assert(quantize_unorm10(1.0f) == 1023);
static_assert(oct_encode16(float3{0, 0, 1}).x == 0 && oct_encode16(float3{0, 0, 1}).y == 0);
static_assert((oct_encode_tangent(float3{0, 0, 1}, 1.0f).x & 1) == 0);  // +1 -> LSB 0
static_assert((oct_encode_tangent(float3{0, 0, 1}, -1.0f).x & 1) == 1); // -1 -> LSB 1

} // namespace vkm
