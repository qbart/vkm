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
#include "common.hpp"
#include "matrix.hpp"
#include "quaternion.hpp"
#include "vector.hpp"

#include <cmath>

namespace vkm {

inline constexpr float pi = 3.14159265358979323846f;

[[nodiscard]] constexpr float rad(float degrees) { return degrees * (pi / 180.0f); }
[[nodiscard]] constexpr float deg(float radians) { return radians * (180.0f / pi); }

// Angle literals: `90.0_deg` / `1.5_rad`, all yielding radians as float so
// signatures stay plain `float` (mirrors Slang/HLSL, which has no angle type).
// `inline namespace` so they resolve unqualified inside vkm; outside, opt in
// with `using namespace vkm::literals;` — literal operators are NOT found by
// ADL, so the suffix is invisible without it. Integer and floating overloads
// are separate: a floating UDL won't match `90_deg`, only `90.0_deg`.
inline namespace literals {
[[nodiscard]] constexpr float operator""_deg(long double d) { return rad(static_cast<float>(d)); }
[[nodiscard]] constexpr float operator""_deg(unsigned long long d) { return rad(static_cast<float>(d)); }
[[nodiscard]] constexpr float operator""_rad(long double r) { return static_cast<float>(r); }
[[nodiscard]] constexpr float operator""_rad(unsigned long long r) { return static_cast<float>(r); }
} // namespace literals

// Euler angles in DEGREES -> quaternion. Convenience over quat_from_euler, which
// takes radians; `angles` = (pitch about X, yaw about Y, roll about Z).
[[nodiscard]] inline quat quat_angles(float3 angles) {
    return quat_from_euler(float3{rad(angles.x), rad(angles.y), rad(angles.z)});
}

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

// Post-rotate `q` by `deg` degrees about `axis` (defaults to world up).
[[nodiscard]] inline quat rotate_deg(quat q, float deg, float3 axis = up) {
    return q * quat::from_axis_angle(axis, rad(deg));
}

// Convenience: T * R * S (scale, then rotate, then translate).
[[nodiscard]] inline float4x4 compose_trs(float3 t, quat r, float3 s) {
    return translate(t) * to_float4x4(r) * scale(s);
}

// ---- vector rotation helper ---------------------------------------------------

// Rotate `current` toward `target`: its direction turns by at most
// max_radians_delta, and its length moves toward |target| by at most
// max_magnitude_delta. Both are clamped against overshoot. (Unity RotateTowards.)
[[nodiscard]] inline float3 rotate_towards(float3 current, float3 target,
                                           float max_radians_delta, float max_magnitude_delta) {
    constexpr float eps = 1e-20f;
    float len_c = length(current);
    float len_t = length(target);
    float new_len = move_towards(len_c, len_t, max_magnitude_delta); // step the magnitude

    // A (near-)zero endpoint has no direction to rotate; keep whatever axis exists.
    if (len_c < eps || len_t < eps) {
        float3 dir = len_c >= eps   ? current / len_c
                     : len_t >= eps ? target / len_t
                                    : float3{0, 0, 0};
        return dir * new_len;
    }

    float3 dc = current / len_c;
    float3 dt = target / len_t;
    float theta = std::acos(clamp(dot(dc, dt), -1.0f, 1.0f)); // angle between dirs, [0, pi]
    if (theta < eps) return dc * new_len;                     // already aligned

    float turn = min(max_radians_delta, theta); // don't overshoot target dir
    float3 axis = cross(dc, dt);
    float axis_len = length(axis);
    if (axis_len < eps) {
        // antiparallel (theta ~ pi): cross is degenerate, pick any perpendicular axis.
        float3 ref = abs(dc.x) < 0.9f ? float3{1, 0, 0} : float3{0, 1, 0};
        axis = normalize(cross(dc, ref));
    } else {
        axis = axis / axis_len;
    }
    return rotate(quat::from_axis_angle(axis, turn), dc) * new_len;
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

// ---- affine TRS decomposition -------------------------------------------------
// Splits an affine/projective 4x4 into translation, rotation, scale (+ skew and a
// perspective row). Orientation is recovered via to_quat once the basis is made
// orthonormal (Gram-Schmidt). Returns false on a degenerate matrix.
inline bool decompose(const float4x4& model, float3& scale, quat& orientation, float3& translation,
                      float3& skew, float4& perspective) {
    constexpr float eps = 1e-6f;
    float4x4 local = model;

    if (vkm::abs(local[3][3]) < eps) return false;

    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) local[c][r] /= local[3][3];

    // Singularity test on the upper-left 3x3 (perspective partition cleared).
    float4x4 persp = local;
    for (int c = 0; c < 3; ++c) persp[c][3] = 0.0f;
    persp[3][3] = 1.0f;
    if (vkm::abs(vkm::determinant(persp)) < eps) return false;

    // Isolate the perspective row (last row, column-major).
    if (vkm::abs(local[0][3]) > eps || vkm::abs(local[1][3]) > eps || vkm::abs(local[2][3]) > eps) {
        float4 rhs{local[0][3], local[1][3], local[2][3], local[3][3]};
        perspective = vkm::transpose(vkm::inverse(persp)) * rhs;
        local[0][3] = 0.0f;
        local[1][3] = 0.0f;
        local[2][3] = 0.0f;
        local[3][3] = 1.0f;
    } else {
        perspective = float4{0, 0, 0, 1};
    }

    translation = float3{local[3][0], local[3][1], local[3][2]};
    local[3] = float4{0, 0, 0, local[3][3]};

    float3 row[3];
    for (int i = 0; i < 3; ++i) row[i] = float3{local[i][0], local[i][1], local[i][2]};

    scale.x = vkm::length(row[0]);
    row[0] = vkm::normalize(row[0]);
    skew.z = vkm::dot(row[0], row[1]);
    row[1] = row[1] - row[0] * skew.z;

    scale.y = vkm::length(row[1]);
    row[1] = vkm::normalize(row[1]);
    skew.z /= scale.y;

    skew.y = vkm::dot(row[0], row[2]);
    row[2] = row[2] - row[0] * skew.y;
    skew.x = vkm::dot(row[1], row[2]);
    row[2] = row[2] - row[1] * skew.x;

    scale.z = vkm::length(row[2]);
    row[2] = vkm::normalize(row[2]);
    skew.y /= scale.z;
    skew.x /= scale.z;

    // Coordinate-system flip: if the basis is left-handed, negate it and the scale.
    if (vkm::dot(row[0], vkm::cross(row[1], row[2])) < 0.0f) {
        for (int i = 0; i < 3; ++i) {
            scale[i] = -scale[i];
            row[i] = -row[i];
        }
    }

    orientation = vkm::to_quat(float3x3{row[0], row[1], row[2]});
    return true;
}

} // namespace vkm
