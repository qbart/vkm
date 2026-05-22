#pragma once
//
// vkm — vector types.
//
// `vector<T, N>` is the one template; `float3`, `uint4`, etc. are aliases, so
// the names mirror Slang/HLSL shader code. Storage is the named components
// themselves (x, y, z, w) — no anonymous unions — which keeps every type
// standard-layout and fully usable in `constexpr`.
//
// Arithmetic is defined once as generic component-wise loops, then `float4`
// gets non-template overloads that drop into the SIMD backend. At compile time
// (`if consteval`) even those fall back to the scalar path, so the whole API is
// constexpr-friendly.
//
#include "detail/simd.hpp"

#include <cmath>
#include <cstdint>

namespace vkm {

using uint = std::uint32_t; // Slang's `uint`

template <class T, int N>
struct vector;

// ---- storage specializations -------------------------------------------------
// Each holds its components directly. operator[] uses a switch so it stays valid
// in a constant expression (pointer arithmetic across members would not be).

template <class T>
struct alignas(2 * sizeof(T)) vector<T, 2> {
    T x, y;
    constexpr vector() = default;
    constexpr explicit vector(T s) : x(s), y(s) {}
    constexpr vector(T x_, T y_) : x(x_), y(y_) {}
    [[nodiscard]] constexpr T& operator[](int i) { return i == 0 ? x : y; }
    [[nodiscard]] constexpr const T& operator[](int i) const { return i == 0 ? x : y; }
    static constexpr int size = 2;
    using value_type = T;
};

template <class T>
struct vector<T, 3> {
    T x, y, z;
    constexpr vector() = default;
    constexpr explicit vector(T s) : x(s), y(s), z(s) {}
    constexpr vector(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    constexpr vector(vector<T, 2> xy, T z_) : x(xy.x), y(xy.y), z(z_) {}
    [[nodiscard]] constexpr T& operator[](int i) { return i == 0 ? x : i == 1 ? y : z; }
    [[nodiscard]] constexpr const T& operator[](int i) const { return i == 0 ? x : i == 1 ? y : z; }
    static constexpr int size = 3;
    using value_type = T;
};

template <class T>
struct alignas(4 * sizeof(T)) vector<T, 4> {
    T x, y, z, w;
    constexpr vector() = default;
    constexpr explicit vector(T s) : x(s), y(s), z(s), w(s) {}
    constexpr vector(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}
    constexpr vector(vector<T, 3> xyz, T w_) : x(xyz.x), y(xyz.y), z(xyz.z), w(w_) {}
    constexpr vector(vector<T, 2> xy, T z_, T w_) : x(xy.x), y(xy.y), z(z_), w(w_) {}
    [[nodiscard]] constexpr T& operator[](int i) { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
    [[nodiscard]] constexpr const T& operator[](int i) const { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
    static constexpr int size = 4;
    using value_type = T;
};

// ---- aliases (Slang/HLSL names) ----------------------------------------------

using float2 = vector<float, 2>;
using float3 = vector<float, 3>;
using float4 = vector<float, 4>;
using double2 = vector<double, 2>;
using double3 = vector<double, 3>;
using double4 = vector<double, 4>;
using int2 = vector<int, 2>;
using int3 = vector<int, 3>;
using int4 = vector<int, 4>;
using uint2 = vector<uint, 2>;
using uint3 = vector<uint, 3>;
using uint4 = vector<uint, 4>;
using bool2 = vector<bool, 2>;
using bool3 = vector<bool, 3>;
using bool4 = vector<bool, 4>;

// ---- generic component-wise arithmetic ---------------------------------------

#define VKM_VEC_BINOP(op)                                                          \
    template <class T, int N>                                                      \
    [[nodiscard]] constexpr vector<T, N> operator op(vector<T, N> a, vector<T, N> b) { \
        vector<T, N> r{};                                                          \
        for (int i = 0; i < N; ++i) r[i] = a[i] op b[i];                           \
        return r;                                                                  \
    }                                                                              \
    template <class T, int N>                                                      \
    [[nodiscard]] constexpr vector<T, N> operator op(vector<T, N> a, T s) {        \
        vector<T, N> r{};                                                          \
        for (int i = 0; i < N; ++i) r[i] = a[i] op s;                              \
        return r;                                                                  \
    }                                                                              \
    template <class T, int N>                                                      \
    [[nodiscard]] constexpr vector<T, N> operator op(T s, vector<T, N> a) {        \
        vector<T, N> r{};                                                          \
        for (int i = 0; i < N; ++i) r[i] = s op a[i];                              \
        return r;                                                                  \
    }

VKM_VEC_BINOP(+)
VKM_VEC_BINOP(-)
VKM_VEC_BINOP(*)
VKM_VEC_BINOP(/)
#undef VKM_VEC_BINOP

template <class T, int N>
[[nodiscard]] constexpr vector<T, N> operator-(vector<T, N> a) {
    vector<T, N> r{};
    for (int i = 0; i < N; ++i) r[i] = -a[i];
    return r;
}

template <class T, int N>
[[nodiscard]] constexpr bool operator==(vector<T, N> a, vector<T, N> b) {
    for (int i = 0; i < N; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

// ---- float4 SIMD overloads ----------------------------------------------------
// Non-template, so they win overload resolution over the generic templates for
// float4 args. `if consteval` keeps them usable in constant expressions.

[[nodiscard]] constexpr float4 operator+(float4 a, float4 b) {
    if consteval { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
    float4 r{};
    detail::store4(&r.x, detail::add(detail::load4(&a.x), detail::load4(&b.x)));
    return r;
}
[[nodiscard]] constexpr float4 operator-(float4 a, float4 b) {
    if consteval { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
    float4 r{};
    detail::store4(&r.x, detail::sub(detail::load4(&a.x), detail::load4(&b.x)));
    return r;
}
[[nodiscard]] constexpr float4 operator*(float4 a, float4 b) {
    if consteval { return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
    float4 r{};
    detail::store4(&r.x, detail::mul(detail::load4(&a.x), detail::load4(&b.x)));
    return r;
}
[[nodiscard]] constexpr float4 operator/(float4 a, float4 b) {
    if consteval { return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w}; }
    float4 r{};
    detail::store4(&r.x, detail::div(detail::load4(&a.x), detail::load4(&b.x)));
    return r;
}
[[nodiscard]] constexpr float4 operator*(float4 a, float s) {
    if consteval { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
    float4 r{};
    detail::store4(&r.x, detail::mul(detail::load4(&a.x), detail::splat(s)));
    return r;
}
[[nodiscard]] constexpr float4 operator*(float s, float4 a) { return a * s; }

// ---- geometric helpers --------------------------------------------------------

template <class T, int N>
[[nodiscard]] constexpr T dot(vector<T, N> a, vector<T, N> b) {
    T acc{};
    for (int i = 0; i < N; ++i) acc += a[i] * b[i];
    return acc;
}

[[nodiscard]] constexpr float dot(float4 a, float4 b) {
    if consteval { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
    return detail::hsum(detail::mul(detail::load4(&a.x), detail::load4(&b.x)));
}

template <class T>
[[nodiscard]] constexpr vector<T, 3> cross(vector<T, 3> a, vector<T, 3> b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

template <class T, int N>
[[nodiscard]] constexpr T length_squared(vector<T, N> v) {
    return dot(v, v);
}

template <class T, int N>
[[nodiscard]] T length(vector<T, N> v) {
    return std::sqrt(dot(v, v));
}

template <class T, int N>
[[nodiscard]] vector<T, N> normalize(vector<T, N> v) {
    return v * (T{1} / length(v));
}

} // namespace vkm
