#pragma once
//
// vkm — quaternion.
//
// Stored as (x, y, z, w) with w the scalar part; identity is (0,0,0,1). Used for
// rotations: composes without gimbal lock and converts cheaply to a rotation
// matrix. Rotations are right-handed and act as v' = q * v (column-vector
// convention, matching the matrices).
//
#include "matrix.hpp"
#include "vector.hpp"

#include <cmath>

namespace vkm {

struct quat {
    float x, y, z, w;

    constexpr quat() : x(0), y(0), z(0), w(1) {} // identity
    constexpr quat(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    [[nodiscard]] static constexpr quat identity() { return {0, 0, 0, 1}; }

    // Rotation of `angle` radians about `axis` (need not be normalized).
    [[nodiscard]] static quat from_axis_angle(float3 axis, float angle) {
        float3 n = normalize(axis);
        float h = angle * 0.5f;
        float s = std::sin(h);
        return {n.x * s, n.y * s, n.z * s, std::cos(h)};
    }

    [[nodiscard]] constexpr float3 xyz() const { return {x, y, z}; }
};

// Hamilton product: applying (a*b) rotates by b, then a.
[[nodiscard]] constexpr quat mul(quat a, quat b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
[[nodiscard]] constexpr quat operator*(quat a, quat b) { return mul(a, b); }

[[nodiscard]] constexpr quat conjugate(quat q) { return {-q.x, -q.y, -q.z, q.w}; }
[[nodiscard]] constexpr float dot(quat a, quat b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
[[nodiscard]] inline float length(quat q) { return std::sqrt(dot(q, q)); }
[[nodiscard]] inline quat normalize(quat q) {
    float inv = 1.0f / length(q);
    return {q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

// Rotate a vector by a unit quaternion (optimized; no matrix needed):
//   t = 2 * cross(q.xyz, v);  v' = v + q.w*t + cross(q.xyz, t)
[[nodiscard]] constexpr float3 rotate(quat q, float3 v) {
    float3 u = q.xyz();
    float3 t = 2.0f * cross(u, v);
    return v + q.w * t + cross(u, t);
}

// Unit quaternion -> 3x3 rotation matrix (column-major, v' = R*v).
[[nodiscard]] constexpr float3x3 to_float3x3(quat q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return float3x3{
        float3{1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)},   // column 0
        float3{2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)},   // column 1
        float3{2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy)}};  // column 2
}

// Unit quaternion -> 4x4 (rotation in the upper-left 3x3).
[[nodiscard]] constexpr float4x4 to_float4x4(quat q) {
    float3x3 r = to_float3x3(q);
    return float4x4{float4{r[0], 0.0f},
                    float4{r[1], 0.0f},
                    float4{r[2], 0.0f},
                    float4{0, 0, 0, 1}};
}

} // namespace vkm
