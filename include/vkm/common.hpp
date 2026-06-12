#pragma once
//
// vkm — common scalar/vector functions, Slang/HLSL spelling.
//
// Names follow Slang: `lerp` (not mix), `frac` (not fract), `saturate`, `rsqrt`.
// Each function has scalar and component-wise vector overloads. The algebraic
// ones (min/max/clamp/lerp/step/smoothstep/abs/sign/reflect) are constexpr; the
// ones that wrap <cmath> (sqrt/floor/sin/...) are runtime-only.
//
#include "vector.hpp"

#include <cmath>
#include <concepts>

namespace vkm {

// ---- min / max ----------------------------------------------------------------

template <class T>
[[nodiscard]] constexpr T min(T a, T b) { return a < b ? a : b; }
template <class T>
[[nodiscard]] constexpr T max(T a, T b) { return a > b ? a : b; }

template <class T, int N>
[[nodiscard]] constexpr vector<T, N> min(vector<T, N> a, vector<T, N> b) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = min(a[i], b[i]);
    return r;
}
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> max(vector<T, N> a, vector<T, N> b) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = max(a[i], b[i]);
    return r;
}
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> min(vector<T, N> a, T s) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = min(a[i], s);
    return r;
}
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> max(vector<T, N> a, T s) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = max(a[i], s);
    return r;
}

// ---- clamp / saturate ---------------------------------------------------------

template <class T>
[[nodiscard]] constexpr T clamp(T x, T lo, T hi) { return max(lo, min(x, hi)); }
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> clamp(vector<T, N> x, T lo, T hi) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = clamp(x[i], lo, hi);
    return r;
}
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> clamp(vector<T, N> x, vector<T, N> lo, vector<T, N> hi) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = clamp(x[i], lo[i], hi[i]);
    return r;
}

template <class T>
[[nodiscard]] constexpr T saturate(T x) { return clamp(x, T{0}, T{1}); }
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> saturate(vector<T, N> x) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = saturate(x[i]);
    return r;
}

// ---- lerp ---------------------------------------------------------------------

template <class T>
[[nodiscard]] constexpr T lerp(T a, T b, T t) { return a + (b - a) * t; }
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> lerp(vector<T, N> a, vector<T, N> b, T t) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = lerp(a[i], b[i], t);
    return r;
}
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> lerp(vector<T, N> a, vector<T, N> b, vector<T, N> t) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = lerp(a[i], b[i], t[i]);
    return r;
}

// ---- step / smoothstep --------------------------------------------------------

template <class T>
[[nodiscard]] constexpr T step(T edge, T x) { return x < edge ? T{0} : T{1}; }
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> step(T edge, vector<T, N> x) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = step(edge, x[i]);
    return r;
}

template <class T>
[[nodiscard]] constexpr T smoothstep(T e0, T e1, T x) {
    T t = saturate((x - e0) / (e1 - e0));
    return t * t * (T{3} - T{2} * t);
}
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> smoothstep(T e0, T e1, vector<T, N> x) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = smoothstep(e0, e1, x[i]);
    return r;
}

// ---- abs / sign ---------------------------------------------------------------

template <class T>
[[nodiscard]] constexpr T abs(T x) { return x < T{0} ? -x : x; }
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> abs(vector<T, N> v) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = abs(v[i]);
    return r;
}
template <class T>
[[nodiscard]] constexpr T sign(T x) { return T(T{0} < x) - T(x < T{0}); }
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> sign(vector<T, N> v) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = sign(v[i]);
    return r;
}

// 1 if a and b are within epsilon of each other, else 0 (HLSL-style mask).
template <class T>
[[nodiscard]] constexpr T close_to(T a, T b, T epsilon) {
    return abs(a - b) <= epsilon ? T{1} : T{0};
}

// ---- geometric ----------------------------------------------------------------

// HLSL reflect: incident i about normal n -> i - 2*dot(i,n)*n.
template <class T, int N>
[[nodiscard]] constexpr vector<T, N> reflect(vector<T, N> i, vector<T, N> n) {
    return i - n * (T{2} * dot(i, n));
}
template <class T, int N>
[[nodiscard]] T distance(vector<T, N> a, vector<T, N> b) { return length(a - b); }
// Squared distance: no sqrt, so it's constexpr and cheaper for comparisons.
template <class T, int N>
[[nodiscard]] constexpr T distance2(vector<T, N> a, vector<T, N> b) {
    return length2(a - b);
}

// A ray as origin + direction. `direction` need not be unit length; the distance
// reported by ray_triangle_intersect is then measured in units of |direction|.
struct ray {
    float3 origin;
    float3 direction;
};

// Moller-Trumbore ray/triangle intersection. On a front- or back-facing hit,
// writes the ray-parameter t to `distance` and returns true; otherwise returns
// false (ray parallel to, or pointing away from, the triangle). The hit point is
// ray.origin + t * ray.direction.
[[nodiscard]] inline bool ray_triangle_intersect(const ray& r, float3 a, float3 b, float3 c,
                                                 float& distance) {
    constexpr float epsilon = 1e-7f;
    float3 edge1 = b - a;
    float3 edge2 = c - a;
    float3 h = cross(r.direction, edge2);
    float det = dot(edge1, h);

    if (det > -epsilon && det < epsilon) return false; // ray parallel to triangle

    float inv_det = 1.0f / det;
    float3 s = r.origin - a;
    float u = dot(s, h) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, edge1);
    float v = dot(r.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = dot(edge2, q) * inv_det;
    if (t > epsilon) {
        distance = t;
        return true;
    }
    return false; // line intersection, but behind the ray origin
}

// ---- projection / angle / move-towards ----------------------------------------

// Unsigned angle between a and b, in RADIANS, range [0, pi]. Inputs need not be
// unit length. Zero if either is ~zero. (Unity Vector3.Angle, but radians.)
template <std::floating_point T, int N>
[[nodiscard]] T angle(vector<T, N> a, vector<T, N> b) {
    T denom = std::sqrt(length2(a) * length2(b)); // |a| * |b|
    if (denom < T(1e-20)) return T{0};
    return std::acos(clamp(dot(a, b) / denom, T{-1}, T{1}));
}

// Projection of v onto onNormal (need not be unit). Zero if onNormal is ~zero.
template <std::floating_point T, int N>
[[nodiscard]] constexpr vector<T, N> project(vector<T, N> v, vector<T, N> on_normal) {
    T sq = dot(on_normal, on_normal);
    if (sq < T(1e-20)) return vector<T, N>{};
    return on_normal * (dot(v, on_normal) / sq);
}

// Component of v in the plane with the given normal: v minus its projection onto
// the normal. Returns v unchanged when planeNormal is ~zero. (Unity ProjectOnPlane.)
template <std::floating_point T, int N>
[[nodiscard]] constexpr vector<T, N> project_on_plane(vector<T, N> v, vector<T, N> plane_normal) {
    return v - project(v, plane_normal);
}

// Step a scalar toward target by at most max_delta. Never overshoots.
template <std::floating_point T>
[[nodiscard]] constexpr T move_towards(T current, T target, T max_delta) {
    return abs(target - current) <= max_delta ? target
                                              : current + sign(target - current) * max_delta;
}

// Move point `current` toward `target` by at most max_distance_delta; never
// overshoots, negative delta moves away. (Unity Vector3.MoveTowards.)
template <std::floating_point T, int N>
[[nodiscard]] vector<T, N> move_towards(vector<T, N> current, vector<T, N> target, T max_distance_delta) {
    vector<T, N> d = target - current;
    T dist = length(d);
    if (dist <= max_distance_delta || dist < T(1e-20)) return target;
    return current + d * (max_distance_delta / dist);
}

// ---- <cmath> wrappers (runtime) -----------------------------------------------

#define VKM_UNARY_STD(name, fn)                                                    \
    template <std::floating_point T>                                               \
    [[nodiscard]] inline T name(T x) { return fn(x); }                             \
    template <std::floating_point T, int N>                                        \
    [[nodiscard]] inline vector<T, N> name(vector<T, N> v) {                        \
        vector<T, N> r{};                                                          \
        for (int i = 0; i < N; ++i) r[i] = fn(v[i]);                               \
        return r;                                                                  \
    }
VKM_UNARY_STD(floor, std::floor)
VKM_UNARY_STD(ceil, std::ceil)
VKM_UNARY_STD(round, std::round)
VKM_UNARY_STD(trunc, std::trunc)
VKM_UNARY_STD(sqrt, std::sqrt)
VKM_UNARY_STD(exp, std::exp)
VKM_UNARY_STD(log, std::log)
VKM_UNARY_STD(exp2, std::exp2)
VKM_UNARY_STD(log2, std::log2)
VKM_UNARY_STD(sin, std::sin)
VKM_UNARY_STD(cos, std::cos)
VKM_UNARY_STD(tan, std::tan)
VKM_UNARY_STD(asin, std::asin)
VKM_UNARY_STD(acos, std::acos)
VKM_UNARY_STD(atan, std::atan)
VKM_UNARY_STD(sinh, std::sinh)
VKM_UNARY_STD(cosh, std::cosh)
VKM_UNARY_STD(tanh, std::tanh)
#undef VKM_UNARY_STD

// ---- binary <cmath> wrappers (runtime) ----------------------------------------

#define VKM_BINARY_STD(name, fn)                                                   \
    template <std::floating_point T>                                               \
    [[nodiscard]] inline T name(T a, T b) { return fn(a, b); }                      \
    template <std::floating_point T, int N>                                        \
    [[nodiscard]] inline vector<T, N> name(vector<T, N> a, vector<T, N> b) {        \
        vector<T, N> r{};                                                          \
        for (int i = 0; i < N; ++i) r[i] = fn(a[i], b[i]);                         \
        return r;                                                                  \
    }
// atan2(y, x): full-circle angle in radians, range (-pi, pi].
VKM_BINARY_STD(atan2, std::atan2)
// fmod(a, b): floating-point remainder, sign follows `a` (like HLSL `fmod`).
VKM_BINARY_STD(fmod, std::fmod)
#undef VKM_BINARY_STD

// Both sine and cosine of x in one call (handy for building rotations).
template <std::floating_point T>
inline void sincos(T x, T& s, T& c) { s = std::sin(x); c = std::cos(x); }

template <std::floating_point T>
[[nodiscard]] inline T frac(T x) { return x - std::floor(x); }
template <std::floating_point T, int N>
[[nodiscard]] inline vector<T, N> frac(vector<T, N> v) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = v[i] - std::floor(v[i]);
    return r;
}

template <std::floating_point T>
[[nodiscard]] inline T rsqrt(T x) { return T{1} / std::sqrt(x); }
template <std::floating_point T, int N>
[[nodiscard]] inline vector<T, N> rsqrt(vector<T, N> v) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = T{1} / std::sqrt(v[i]);
    return r;
}

template <std::floating_point T>
[[nodiscard]] inline T pow(T a, T b) { return std::pow(a, b); }
template <std::floating_point T, int N>
[[nodiscard]] inline vector<T, N> pow(vector<T, N> a, vector<T, N> b) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = std::pow(a[i], b[i]);
    return r;
}

// ---- game-loop helpers --------------------------------------------------------

// Wrap x into [0, length) — like a positive fmod (Unity Mathf.Repeat). Useful
// for looping animation time or wrapping a value around a range.
template <std::floating_point T>
[[nodiscard]] inline T repeat(T x, T length) {
    return clamp(x - std::floor(x / length) * length, T{0}, length);
}

// Bounce x back and forth in [0, length] at unit speed (Unity Mathf.PingPong).
template <std::floating_point T>
[[nodiscard]] inline T ping_pong(T x, T length) {
    T t = repeat(x, length * T{2});
    return length - abs(t - length);
}

// Shortest-path angular lerp, in RADIANS. Interpolates the wrapped difference so
// it crosses the +-pi seam correctly (Unity Mathf.LerpAngle, but radians).
template <std::floating_point T>
[[nodiscard]] inline T lerp_angle(T a, T b, T t) {
    constexpr T two_pi = T{6.283185307179586476925286766559};
    T delta = repeat(b - a, two_pi);
    if (delta > two_pi / T{2}) delta -= two_pi;
    return a + delta * t;
}

} // namespace vkm
