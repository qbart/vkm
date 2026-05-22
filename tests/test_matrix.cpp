// vkm — matrix smoke tests.
//
// Pins down the column-major + M*v convention: a translation lives in the last
// COLUMN, M*v is a linear combination of columns, and composition is
// right-to-left.

#include <vkm/vkm.hpp>

#include <cmath>
#include <cstdio>

using namespace vkm;

static int g_failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                \
        }                                                                \
    } while (0)

// Column-major translation: translation components sit in the last column.
static constexpr float4x4 translation(float x, float y, float z) {
    return float4x4{float4{1, 0, 0, 0},
                    float4{0, 1, 0, 0},
                    float4{0, 0, 1, 0},
                    float4{x, y, z, 1}};
}

static constexpr float4x4 scaling(float x, float y, float z) {
    return float4x4{float4{x, 0, 0, 0},
                    float4{0, y, 0, 0},
                    float4{0, 0, z, 0},
                    float4{0, 0, 0, 1}};
}

// ---- compile-time --------------------------------------------------------------

static_assert(float4x4::identity() * float4{1, 2, 3, 4} == float4{1, 2, 3, 4});
static_assert(translation(10, 20, 30) * float4{1, 2, 3, 1} == float4{11, 22, 33, 1});
static_assert(scaling(2, 3, 4) * float4{1, 1, 1, 1} == float4{2, 3, 4, 1});
// Right-to-left composition: scale THEN translate == (T * S).
static_assert((translation(10, 0, 0) * scaling(2, 1, 1)) * float4{1, 0, 0, 1} == float4{12, 0, 0, 1});
// Element access: translation x sits at row 0, last column.
static_assert(translation(7, 8, 9)[0, 3] == 7.0f);
static_assert(translation(7, 8, 9)[3] == float4{7, 8, 9, 1}); // m[c] is a column

int main() {
    float4x4 T = translation(10, 20, 30);
    float4x4 S = scaling(2, 3, 4);

    // M * v at runtime (float4 SIMD path under the hood).
    CHECK((T * float4{1, 2, 3, 1}) == float4(11, 22, 33, 1));
    CHECK((S * float4{1, 1, 1, 1}) == float4(2, 3, 4, 1));

    // Compose: scale then translate, applied to a point.
    float4x4 TS = T * S;
    float4 p{1, 1, 1, 1};
    CHECK((TS * p) == (T * (S * p)));
    CHECK((TS * p) == float4(12, 23, 34, 1));

    // transpose moves the translation row -> column.
    float4x4 Tt = transpose(T);
    CHECK((Tt[0, 3]) == 0.0f);
    CHECK((Tt[3, 0]) == 10.0f);

    // Identity is a left/right multiply no-op.
    CHECK((float4x4::identity() * T) == T);

    if (g_failures == 0) {
        std::puts("vkm matrix tests passed");
        return 0;
    }
    std::printf("vkm matrix tests: %d failure(s)\n", g_failures);
    return 1;
}
