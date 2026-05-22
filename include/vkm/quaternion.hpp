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

    // Member-style API (PascalCase). Implemented directly so the type stays
    // self-contained; free-function forms also exist.
    [[nodiscard]] float Len() const { return std::sqrt(x * x + y * y + z * z + w * w); }
    quat& Normalize() { // in place
        float inv = 1.0f / Len();
        x *= inv; y *= inv; z *= inv; w *= inv;
        return *this;
    }
    [[nodiscard]] constexpr quat Conjugate() const { return {-x, -y, -z, w}; }
    [[nodiscard]] constexpr quat Inverse() const { // conjugate / |q|^2
        float inv = 1.0f / (x * x + y * y + z * z + w * w);
        return {-x * inv, -y * inv, -z * inv, w * inv};
    }
    quat& Invert() { return *this = Inverse(); } // in place
    [[nodiscard]] quat Normalized() const {      // unit copy
        float inv = 1.0f / Len();
        return {x * inv, y * inv, z * inv, w * inv};
    }
    // Mutating Unity-style setters (Unity's instance methods). Defined out-of-line
    // below, after the free builders they delegate to.
    quat& SetFromToRotation(float3 from_direction, float3 to_direction);
    quat& SetLookRotation(float3 view, float3 up_hint = up);
    [[nodiscard]] float* Ptr() { return &x; }
    [[nodiscard]] const float* Ptr() const { return &x; }
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

[[nodiscard]] constexpr bool operator==(quat a, quat b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
constexpr quat& operator*=(quat& a, quat b) { return a = a * b; } // in-place Hamilton product
[[nodiscard]] constexpr quat inverse(quat q) {                    // conjugate / |q|^2
    float inv = 1.0f / dot(q, q);
    return {-q.x * inv, -q.y * inv, -q.z * inv, q.w * inv};
}

// ---- interpolation ------------------------------------------------------------
// Raw lerp does NOT preserve unit length, so it's rarely what you want for a
// rotation on its own. Reach for nlerp (cheap; great for blending close
// orientations) or slerp (constant angular velocity; correct for wide arcs).

// Component-wise linear interpolation (unnormalized).
[[nodiscard]] constexpr quat lerp(quat a, quat b, float t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
}

// Normalized lerp. Picks the shortest arc (flips b if the dot is negative).
[[nodiscard]] inline quat nlerp(quat a, quat b, float t) {
    if (dot(a, b) < 0.0f) b = {-b.x, -b.y, -b.z, -b.w};
    return normalize(lerp(a, b, t));
}

// Spherical linear interpolation: constant angular velocity along the shortest
// arc. Falls back to nlerp when the inputs are nearly parallel (sin θ → 0).
[[nodiscard]] inline quat slerp(quat a, quat b, float t) {
    float cos_theta = dot(a, b);
    if (cos_theta < 0.0f) { // shortest path
        b = {-b.x, -b.y, -b.z, -b.w};
        cos_theta = -cos_theta;
    }
    if (cos_theta > 0.9995f) return nlerp(a, b, t); // avoid division by ~0
    float theta = std::acos(cos_theta);
    float inv_sin = 1.0f / std::sin(theta);
    float wa = std::sin((1.0f - t) * theta) * inv_sin;
    float wb = std::sin(t * theta) * inv_sin;
    return {wa * a.x + wb * b.x, wa * a.y + wb * b.y,
            wa * a.z + wb * b.z, wa * a.w + wb * b.w};
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

// Rotation matrix -> unit quaternion (inverse of to_float3x3). Shepperd's method:
// build from whichever diagonal term is largest, for numerical stability.
[[nodiscard]] inline quat to_quat(float3x3 m) {
    // Column-major: m[c][r] is the element at row r, column c.
    float m00 = m[0][0], m11 = m[1][1], m22 = m[2][2];
    float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        float s = std::sqrt(trace + 1.0f) * 2.0f; // 4*w
        return {(m[1][2] - m[2][1]) / s, (m[2][0] - m[0][2]) / s,
                (m[0][1] - m[1][0]) / s, 0.25f * s};
    } else if (m00 > m11 && m00 > m22) {
        float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f; // 4*x
        return {0.25f * s, (m[1][0] + m[0][1]) / s,
                (m[2][0] + m[0][2]) / s, (m[1][2] - m[2][1]) / s};
    } else if (m11 > m22) {
        float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f; // 4*y
        return {(m[1][0] + m[0][1]) / s, 0.25f * s,
                (m[2][1] + m[1][2]) / s, (m[2][0] - m[0][2]) / s};
    } else {
        float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f; // 4*z
        return {(m[2][0] + m[0][2]) / s, (m[2][1] + m[1][2]) / s,
                0.25f * s, (m[0][1] - m[1][0]) / s};
    }
}

// ---- Unity-style queries / construction (radians) -----------------------------

// Angular difference between two unit rotations, in RADIANS, range [0, pi].
// (Unity Quaternion.Angle, but radians.) Measured from the relative rotation via
// atan2, which stays accurate near 0 and pi where 2*acos(|dot|) loses precision.
[[nodiscard]] inline float angle(quat a, quat b) {
    quat r = conjugate(a) * b;                          // relative rotation (a assumed unit)
    float im = std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z); // |sin(theta/2)|
    return 2.0f * std::atan2(im, std::fabs(r.w));
}

// Shortest-arc rotation taking direction `from` onto direction `to` (neither need
// be unit). (Unity Quaternion.FromToRotation.)
[[nodiscard]] inline quat from_to_rotation(float3 from, float3 to) {
    float3 f = normalize(from);
    float3 t = normalize(to);
    float d = dot(f, t);
    if (d >= 1.0f - 1e-6f) return quat::identity(); // already aligned
    if (d <= -1.0f + 1e-6f) {                        // opposite: 180° about any ⟂ axis
        float3 axis = cross(right, f);
        if (length2(axis) < 1e-6f) axis = cross(up, f);
        axis = normalize(axis);
        return {axis.x, axis.y, axis.z, 0.0f}; // 180° -> w = cos(90°) = 0
    }
    float3 c = cross(f, t);
    float s = std::sqrt((1.0f + d) * 2.0f);
    float inv = 1.0f / s;
    return normalize(quat{c.x * inv, c.y * inv, c.z * inv, s * 0.5f});
}

// Rotation whose forward (local -Z, vkm::forward) points along `forward`, with
// local up aligned as closely as possible to `up_hint`. RH / Y-up — see the
// direction constants in vector.hpp. (Unity Quaternion.LookRotation; Unity is
// left-handed with +Z forward, so its result differs by that convention.)
[[nodiscard]] inline quat look_rotation(float3 forward, float3 up_hint = up) {
    float3 f = normalize(forward);
    float3 r = normalize(cross(f, up_hint)); // right
    float3 u = cross(r, f);                  // re-orthogonalized up
    // Columns map local +X,+Y,+Z to world. Local -Z is forward, so column 2
    // (image of local +Z) is -f.
    return to_quat(float3x3{r, u, -f});
}

// Rotate quaternion `from` toward `to` by at most max_radians_delta; snaps to
// `to` once within reach. (Unity Quaternion.RotateTowards, but radians.)
[[nodiscard]] inline quat rotate_towards(quat from, quat to, float max_radians_delta) {
    float a = angle(from, to);
    if (a < 1e-6f) return to;
    float t = max_radians_delta / a;
    return t >= 1.0f ? to : slerp(from, to, t);
}

inline quat& quat::SetFromToRotation(float3 from_direction, float3 to_direction) {
    return *this = from_to_rotation(from_direction, to_direction);
}
inline quat& quat::SetLookRotation(float3 view, float3 up_hint) {
    return *this = look_rotation(view, up_hint);
}

} // namespace vkm
