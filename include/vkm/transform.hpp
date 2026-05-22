#pragma once
//
// vkm — affine transforms and Vulkan projections.
//
// Conventions (see DESIGN.md):
//   * column-major matrices, column-vector math: v' = M * v
//   * view space is right-handed, camera looks down -Z
//   * VULKAN clip space: depth in [0, 1] (not GL's [-1, 1]) and Y points DOWN.
//     The Y-down flip is baked into the projection builders (proj[1][1] *= -1),
//     so a vkm projection is upload-and-go for a default Vulkan viewport.
//
#include "matrix.hpp"
#include "quaternion.hpp"
#include "vector.hpp"

#include <cmath>

namespace vkm {

inline constexpr float pi = 3.14159265358979323846f;

[[nodiscard]] constexpr float radians(float degrees) { return degrees * (pi / 180.0f); }
[[nodiscard]] constexpr float degrees(float radians) { return radians * (180.0f / pi); }

// ---- affine builders ----------------------------------------------------------

// Translation lives in the last column (column-major).
[[nodiscard]] constexpr float4x4 translate(float3 t) {
    return float4x4{float4{1, 0, 0, 0},
                    float4{0, 1, 0, 0},
                    float4{0, 0, 1, 0},
                    float4{t.x, t.y, t.z, 1}};
}

[[nodiscard]] constexpr float4x4 scale(float3 s) {
    return float4x4{float4{s.x, 0, 0, 0},
                    float4{0, s.y, 0, 0},
                    float4{0, 0, s.z, 0},
                    float4{0, 0, 0, 1}};
}

[[nodiscard]] inline float4x4 rotate(float angle, float3 axis) {
    return to_float4x4(quat::from_axis_angle(axis, angle));
}

// Convenience: T * R * S (scale, then rotate, then translate).
[[nodiscard]] inline float4x4 compose_trs(float3 t, quat r, float3 s) {
    return translate(t) * to_float4x4(r) * scale(s);
}

// ---- view ---------------------------------------------------------------------

// Right-handed look-at: camera at `eye` looking toward `center`, world up `up`.
[[nodiscard]] inline float4x4 look_at(float3 eye, float3 center, float3 up) {
    float3 f = normalize(center - eye); // forward
    float3 s = normalize(cross(f, up)); // right
    float3 u = cross(s, f);             // recomputed up
    return float4x4{float4{s.x, u.x, -f.x, 0},
                    float4{s.y, u.y, -f.y, 0},
                    float4{s.z, u.z, -f.z, 0},
                    float4{-dot(s, eye), -dot(u, eye), dot(f, eye), 1}};
}

// ---- projection (Vulkan: depth [0,1], Y down) --------------------------------

// Right-handed perspective. `fovy` is the vertical field of view in radians.
[[nodiscard]] inline float4x4 perspective(float fovy, float aspect, float near, float far) {
    float t = std::tan(fovy * 0.5f);
    float4x4 m; // zero
    m[0, 0] = 1.0f / (aspect * t);
    m[1, 1] = -1.0f / t; // Y-flip for Vulkan
    m[2, 2] = far / (near - far);
    m[3, 2] = -1.0f;
    m[2, 3] = (far * near) / (near - far);
    return m;
}

// Right-handed orthographic.
[[nodiscard]] constexpr float4x4 ortho(float left, float right, float bottom, float top,
                                       float near, float far) {
    float4x4 m; // zero
    m[0, 0] = 2.0f / (right - left);
    m[1, 1] = -2.0f / (top - bottom); // Y-flip for Vulkan
    m[2, 2] = -1.0f / (far - near);
    m[0, 3] = -(right + left) / (right - left);
    m[1, 3] = (top + bottom) / (top - bottom); // Y-flip: row negated
    m[2, 3] = -near / (far - near);
    m[3, 3] = 1.0f;
    return m;
}

} // namespace vkm
