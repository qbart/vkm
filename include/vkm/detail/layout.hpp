#pragma once
//
// vkm — std140 / std430 layout helpers for UBO/SSBO interop.
//
// GPU buffer blocks follow strict alignment rules that differ from C++'s. The
// infamous one: a 3-component vector has a base alignment of 16 (not 12), in
// BOTH std140 and std430. Get this wrong and every field past the first vec3 is
// silently misaligned against the shader.
//
// This header gives you two things:
//
//   1. Alignment traits — std140_align_v<T> / std430_align_v<T> — the rules as
//      constexpr values. Use them with `alignas` on members for byte-exact
//      packing, INCLUDING the vec3+scalar "tuck-in" the spec allows.
//
//   2. Wrapper types — std140<T> / std430<T> and std140_array/std430_array —
//      that carry the right alignment (and, for matrices, pad each column).
//      Ergonomic and composable; the tradeoff is each wrapped member gets its
//      own slot, so a wrapped vec3 occupies 16 bytes (no tuck-in). Pick the
//      wrapper path and declare your Slang/GLSL struct the same way, OR use the
//      alignas-trait path for literal byte-for-byte parity.
//
// Rules encoded (base alignment):
//   scalar (float/int/uint)        : 4        (double: 8)
//   vec2                           : 2 * sizeof(component)
//   vec3, vec4                     : 4 * sizeof(component)
//   matrix                         : array of columns; each column is a vecR
//   array element / matrix column  : std140 rounds the slot up to 16; std430
//                                    keeps the natural alignment.
//   bool is intentionally unsupported — GLSL bool is 4 bytes, C++ bool is 1.
//   Use uint for GPU-facing booleans.
//
#include "../matrix.hpp"
#include "../vector.hpp"

#include <cstddef>

namespace vkm {

// ---- alignment traits ---------------------------------------------------------

// Primary template handles scalars (float/double/int/uint): base align = sizeof.
template <class T>
struct std140_layout {
    static constexpr std::size_t align = sizeof(T);
};
template <class T, int N>
struct std140_layout<vector<T, N>> {
    static constexpr std::size_t align = (N == 2 ? 2 : 4) * sizeof(T);
};
template <class T, int R, int C>
struct std140_layout<matrix<T, R, C>> {
    static constexpr std::size_t align = 16; // array-of-columns rounds to 16
};

template <class T>
struct std430_layout {
    static constexpr std::size_t align = sizeof(T);
};
template <class T, int N>
struct std430_layout<vector<T, N>> {
    static constexpr std::size_t align = (N == 2 ? 2 : 4) * sizeof(T);
};
template <class T, int R, int C>
struct std430_layout<matrix<T, R, C>> {
    static constexpr std::size_t align = std430_layout<vector<T, R>>::align; // column align, not rounded
};

template <class T>
inline constexpr std::size_t std140_align_v = std140_layout<T>::align;
template <class T>
inline constexpr std::size_t std430_align_v = std430_layout<T>::align;

// ---- scalar / vector wrappers -------------------------------------------------

template <class T>
struct alignas(std140_align_v<T>) std140 {
    T value{};
    constexpr std140() = default;
    constexpr std140(const T& v) : value(v) {}
    constexpr operator T() const { return value; }
    constexpr std140& operator=(const T& v) {
        value = v;
        return *this;
    }
    [[nodiscard]] constexpr T& get() { return value; }
    [[nodiscard]] constexpr const T& get() const { return value; }
};

template <class T>
struct alignas(std430_align_v<T>) std430 {
    T value{};
    constexpr std430() = default;
    constexpr std430(const T& v) : value(v) {}
    constexpr operator T() const { return value; }
    constexpr std430& operator=(const T& v) {
        value = v;
        return *this;
    }
    [[nodiscard]] constexpr T& get() { return value; }
    [[nodiscard]] constexpr const T& get() const { return value; }
};

// ---- matrix wrappers (pad each column to its slot) ----------------------------

template <class T, int R, int C>
struct alignas(16) std140<matrix<T, R, C>> {
    struct alignas(16) column {
        vector<T, R> v{};
    };
    column columns[C];
    constexpr std140() = default;
    constexpr std140(const matrix<T, R, C>& m) {
        for (int c = 0; c < C; ++c) columns[c].v = m[c];
    }
    constexpr operator matrix<T, R, C>() const {
        matrix<T, R, C> m;
        for (int c = 0; c < C; ++c) m[c] = columns[c].v;
        return m;
    }
};

template <class T, int R, int C>
struct alignas(std430_align_v<vector<T, R>>) std430<matrix<T, R, C>> {
    struct alignas(std430_align_v<vector<T, R>>) column {
        vector<T, R> v{};
    };
    column columns[C];
    constexpr std430() = default;
    constexpr std430(const matrix<T, R, C>& m) {
        for (int c = 0; c < C; ++c) columns[c].v = m[c];
    }
    constexpr operator matrix<T, R, C>() const {
        matrix<T, R, C> m;
        for (int c = 0; c < C; ++c) m[c] = columns[c].v;
        return m;
    }
};

// ---- arrays -------------------------------------------------------------------
// std140: every element slot is rounded up to 16. std430: natural element align.

template <class T, std::size_t N>
struct alignas(16) std140_array {
    struct alignas(16) element {
        std140<T> v{};
    };
    element data[N];
    [[nodiscard]] constexpr std140<T>& operator[](std::size_t i) { return data[i].v; }
    [[nodiscard]] constexpr const std140<T>& operator[](std::size_t i) const { return data[i].v; }
    [[nodiscard]] static constexpr std::size_t size() { return N; }
};

template <class T, std::size_t N>
struct alignas(std430_align_v<T>) std430_array {
    struct alignas(std430_align_v<T>) element {
        std430<T> v{};
    };
    element data[N];
    [[nodiscard]] constexpr std430<T>& operator[](std::size_t i) { return data[i].v; }
    [[nodiscard]] constexpr const std430<T>& operator[](std::size_t i) const { return data[i].v; }
    [[nodiscard]] static constexpr std::size_t size() { return N; }
};

} // namespace vkm
