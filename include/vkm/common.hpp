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
VKM_UNARY_STD(sin, std::sin)
VKM_UNARY_STD(cos, std::cos)
VKM_UNARY_STD(tan, std::tan)
#undef VKM_UNARY_STD

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

} // namespace vkm
