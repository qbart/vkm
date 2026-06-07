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

// Forward declarations: the members transposed()/transpose()/inverse()/invert()/
// determinant() delegate to these free forms (defined below the struct). Member
// and free share a name, so the bodies qualify the call (vkm::transpose, ...) and
// qualified lookup needs the names visible before the struct.
template <class T, int R, int C>
struct matrix;
template <class T, int R, int C> [[nodiscard]] constexpr matrix<T, C, R> transpose(const matrix<T, R, C>& m);
template <class T> [[nodiscard]] constexpr matrix<T, 2, 2> inverse(const matrix<T, 2, 2>& m);
template <class T> [[nodiscard]] constexpr matrix<T, 3, 3> inverse(const matrix<T, 3, 3>& m);
template <class T> [[nodiscard]] constexpr matrix<T, 4, 4> inverse(const matrix<T, 4, 4>& m);
template <class T> [[nodiscard]] constexpr T determinant(const matrix<T, 2, 2>& m);
template <class T> [[nodiscard]] constexpr T determinant(const matrix<T, 3, 3>& m);
template <class T> [[nodiscard]] constexpr T determinant(const matrix<T, 4, 4>& m);

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

    // Member-style API (camelCase, fluent). Free-function forms also exist.
    // Column-major storage, so ptr() is ready to memcpy into a UBO.
    [[nodiscard]] T* ptr() { return &cols[0].x; }
    [[nodiscard]] const T* ptr() const { return &cols[0].x; }
    [[nodiscard]] constexpr matrix<T, C, R> transposed() const { return vkm::transpose(*this); }
    constexpr matrix& transpose()
        requires(R == C)
    { return *this = vkm::transpose(*this); } // in place
    [[nodiscard]] constexpr matrix inverse() const
        requires(R == C && R >= 2 && R <= 4)
    { return vkm::inverse(*this); }
    constexpr matrix& invert()
        requires(R == C && R >= 2 && R <= 4)
    { return *this = vkm::inverse(*this); } // in place
    [[nodiscard]] constexpr T determinant() const
        requires(R == C && R >= 2 && R <= 4)
    { return vkm::determinant(*this); }

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

// GLSL-style aliases. SQUARE matrices only: GLSL names non-square matrices
// columns-first (GLSL mat3x4 = 3 cols x 4 rows) — the opposite of Slang's
// rows-first floatRxC — so aliasing those would invite mistakes. For non-square,
// use the floatRxC names directly.
using mat2 = float2x2;
using mat3 = float3x3;
using mat4 = float4x4;
using dmat2 = double2x2;
using dmat3 = double3x3;
using dmat4 = double4x4;

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

// ---- compound assignment ------------------------------------------------------

template <class T, int R, int C>
constexpr matrix<T, R, C>& operator+=(matrix<T, R, C>& a, const matrix<T, R, C>& b) { return a = a + b; }
template <class T, int R, int C>
constexpr matrix<T, R, C>& operator-=(matrix<T, R, C>& a, const matrix<T, R, C>& b) { return a = a - b; }
template <class T, int R, int C>
constexpr matrix<T, R, C>& operator*=(matrix<T, R, C>& a, T s) { return a = a * s; }
template <class T, int N> // square only: result keeps the same type
constexpr matrix<T, N, N>& operator*=(matrix<T, N, N>& a, const matrix<T, N, N>& b) { return a = a * b; }

// ---- determinant / inverse (square float/double matrices) --------------------
// e(r, c) reads the element at row r, column c. inverse() assumes the matrix is
// invertible; a singular matrix yields inf/nan (as in GLM), not a thrown error.

template <class T>
[[nodiscard]] constexpr T determinant(const matrix<T, 2, 2>& m) {
    return m.cols[0][0] * m.cols[1][1] - m.cols[1][0] * m.cols[0][1];
}
template <class T>
[[nodiscard]] constexpr matrix<T, 2, 2> inverse(const matrix<T, 2, 2>& m) {
    T a = m.cols[0][0], c = m.cols[0][1], b = m.cols[1][0], d = m.cols[1][1];
    T inv = T{1} / (a * d - b * c);
    return matrix<T, 2, 2>{vector<T, 2>{d * inv, -c * inv}, vector<T, 2>{-b * inv, a * inv}};
}

template <class T>
[[nodiscard]] constexpr T determinant(const matrix<T, 3, 3>& m) {
    auto e = [&](int r, int col) { return m.cols[col][r]; };
    return e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
         - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
         + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0));
}
template <class T>
[[nodiscard]] constexpr matrix<T, 3, 3> inverse(const matrix<T, 3, 3>& m) {
    auto e = [&](int r, int col) { return m.cols[col][r]; };
    T a = e(0, 0), b = e(0, 1), c = e(0, 2);
    T d = e(1, 0), f = e(1, 1), g = e(1, 2);
    T h = e(2, 0), i = e(2, 1), j = e(2, 2);
    T inv = T{1} / determinant(m);
    // Columns of the result (adjugate / det).
    return matrix<T, 3, 3>{
        vector<T, 3>{(f * j - g * i) * inv, (g * h - d * j) * inv, (d * i - f * h) * inv},
        vector<T, 3>{(c * i - b * j) * inv, (a * j - c * h) * inv, (b * h - a * i) * inv},
        vector<T, 3>{(b * g - c * f) * inv, (c * d - a * g) * inv, (a * f - b * d) * inv}};
}

template <class T>
[[nodiscard]] constexpr matrix<T, 4, 4> inverse(const matrix<T, 4, 4>& src) {
    // Copy to a flat column-major array (m[c*4 + r] == element row r, col c).
    // A local array (not a reinterpreted pointer) keeps this valid in constexpr.
    T m[16];
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) m[c * 4 + r] = src.cols[c][r];
    T inv[16];
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

    T det = T{1} / (m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12]);
    matrix<T, 4, 4> out;
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) out.cols[c][r] = inv[c * 4 + r] * det;
    return out;
}

template <class T>
[[nodiscard]] constexpr T determinant(const matrix<T, 4, 4>& src) {
    T m[16];
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) m[c * 4 + r] = src.cols[c][r];
    T c0 =  m[5]*(m[10]*m[15]-m[11]*m[14]) - m[9]*(m[6]*m[15]-m[7]*m[14]) + m[13]*(m[6]*m[11]-m[7]*m[10]);
    T c1 = -m[4]*(m[10]*m[15]-m[11]*m[14]) + m[8]*(m[6]*m[15]-m[7]*m[14]) - m[12]*(m[6]*m[11]-m[7]*m[10]);
    T c2 =  m[4]*(m[9]*m[15]-m[11]*m[13])  - m[8]*(m[5]*m[15]-m[7]*m[13]) + m[12]*(m[5]*m[11]-m[7]*m[9]);
    T c3 = -m[4]*(m[9]*m[14]-m[10]*m[13])  + m[8]*(m[5]*m[14]-m[6]*m[13]) - m[12]*(m[5]*m[10]-m[6]*m[9]);
    return m[0] * c0 + m[1] * c1 + m[2] * c2 + m[3] * c3;
}

// Normal matrix: transpose(inverse(upper-left 3x3)) of a model/model-view matrix.
// Use it to transform normals when the model has non-uniform scale.
template <class T>
[[nodiscard]] constexpr matrix<T, 3, 3> normal_matrix(const matrix<T, 4, 4>& m) {
    matrix<T, 3, 3> upper{m.cols[0].xyz(), m.cols[1].xyz(), m.cols[2].xyz()};
    return transpose(inverse(upper));
}

} // namespace vkm
