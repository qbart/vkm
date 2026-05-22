#pragma once
//
// vkm — matrix types.
//
// `matrix<T, R, C>` = R rows, C columns (Slang's `floatRxC` reading). Storage is
// **column-major**: an array of C column vectors. That choice means:
//
//   * a matrix uploads to a Vulkan UBO/SSBO with a plain memcpy (SPIR-V's default
//     matrix layout is column-major) — no transpose, and
//   * `M * v` is a linear combination of the columns, so it falls straight onto
//     the float4 SIMD operators with no extra code.
//
// Math convention is column-vector: v' = M * v, and transforms compose
// right-to-left, M = P * V * Model.
//
// Indexing (C++23 multidimensional subscript):
//   m[c]      -> column c, a vector<T,R>   (note: GLM-like, NOT HLSL's row)
//   m[r, c]   -> the element at row r, column c
//
#include "vector.hpp"

#include <concepts>

namespace vkm {

template <class T, int R, int C>
struct matrix {
    vector<T, R> cols[C]; // column-major: cols[c] is the c-th column

    constexpr matrix() : cols{} {} // zero

    // Construct from C columns: float4x4(c0, c1, c2, c3).
    template <class... Cs>
        requires(sizeof...(Cs) == C) && (std::convertible_to<Cs, vector<T, R>> && ...)
    constexpr explicit matrix(Cs... c) : cols{vector<T, R>(c)...} {}

    // m[c] -> column c.
    [[nodiscard]] constexpr vector<T, R>& operator[](int c) { return cols[c]; }
    [[nodiscard]] constexpr const vector<T, R>& operator[](int c) const { return cols[c]; }

    // m[r, c] -> element at row r, column c (C++23 multidimensional subscript).
    [[nodiscard]] constexpr T& operator[](int r, int c) { return cols[c][r]; }
    [[nodiscard]] constexpr const T& operator[](int r, int c) const { return cols[c][r]; }

    [[nodiscard]] constexpr vector<T, R> col(int c) const { return cols[c]; }
    [[nodiscard]] constexpr vector<T, C> row(int r) const {
        vector<T, C> out{};
        for (int c = 0; c < C; ++c) out[c] = cols[c][r];
        return out;
    }

    // Identity (square only).
    [[nodiscard]] static constexpr matrix identity()
        requires(R == C)
    {
        matrix m;
        for (int i = 0; i < R; ++i) m.cols[i][i] = T{1};
        return m;
    }

    static constexpr int rows = R;
    static constexpr int columns = C;
    using value_type = T;
};

// ---- aliases (Slang/HLSL names: floatRxC = R rows x C cols) -------------------

using float2x2 = matrix<float, 2, 2>;
using float3x3 = matrix<float, 3, 3>;
using float4x4 = matrix<float, 4, 4>;
using float2x3 = matrix<float, 2, 3>;
using float3x2 = matrix<float, 3, 2>;
using float3x4 = matrix<float, 3, 4>;
using float4x3 = matrix<float, 4, 3>;
using double2x2 = matrix<double, 2, 2>;
using double3x3 = matrix<double, 3, 3>;
using double4x4 = matrix<double, 4, 4>;

// ---- linear algebra -----------------------------------------------------------

// M * v: linear combination of columns. For float4x4*float4 this is float4
// scalar-multiply + add, i.e. the SIMD path, with no special-casing here.
template <class T, int R, int C>
[[nodiscard]] constexpr vector<T, R> mul(const matrix<T, R, C>& m, const vector<T, C>& v) {
    vector<T, R> r = m.cols[0] * v[0];
    for (int c = 1; c < C; ++c) r = r + m.cols[c] * v[c];
    return r;
}

// M(RxK) * N(KxC) = (RxC). Each result column is M * (column of N).
template <class T, int R, int K, int C>
[[nodiscard]] constexpr matrix<T, R, C> mul(const matrix<T, R, K>& a, const matrix<T, K, C>& b) {
    matrix<T, R, C> out;
    for (int c = 0; c < C; ++c) out.cols[c] = mul(a, b.cols[c]);
    return out;
}

template <class T, int R, int K, int C>
[[nodiscard]] constexpr matrix<T, R, C> operator*(const matrix<T, R, K>& a, const matrix<T, K, C>& b) {
    return mul(a, b);
}
template <class T, int R, int C>
[[nodiscard]] constexpr vector<T, R> operator*(const matrix<T, R, C>& m, const vector<T, C>& v) {
    return mul(m, v);
}

template <class T, int R, int C>
[[nodiscard]] constexpr matrix<T, C, R> transpose(const matrix<T, R, C>& m) {
    matrix<T, C, R> out;
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) out.cols[r][c] = m.cols[c][r];
    return out;
}

template <class T, int R, int C>
[[nodiscard]] constexpr matrix<T, R, C> operator+(const matrix<T, R, C>& a, const matrix<T, R, C>& b) {
    matrix<T, R, C> out;
    for (int c = 0; c < C; ++c) out.cols[c] = a.cols[c] + b.cols[c];
    return out;
}
template <class T, int R, int C>
[[nodiscard]] constexpr matrix<T, R, C> operator-(const matrix<T, R, C>& a, const matrix<T, R, C>& b) {
    matrix<T, R, C> out;
    for (int c = 0; c < C; ++c) out.cols[c] = a.cols[c] - b.cols[c];
    return out;
}
template <class T, int R, int C>
[[nodiscard]] constexpr matrix<T, R, C> operator*(const matrix<T, R, C>& m, T s) {
    matrix<T, R, C> out;
    for (int c = 0; c < C; ++c) out.cols[c] = m.cols[c] * s;
    return out;
}
template <class T, int R, int C>
[[nodiscard]] constexpr matrix<T, R, C> operator*(T s, const matrix<T, R, C>& m) {
    return m * s;
}

template <class T, int R, int C>
[[nodiscard]] constexpr bool operator==(const matrix<T, R, C>& a, const matrix<T, R, C>& b) {
    for (int c = 0; c < C; ++c)
        if (!(a.cols[c] == b.cols[c])) return false;
    return true;
}

} // namespace vkm
