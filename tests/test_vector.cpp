// vkm — vector smoke tests.
//
// Covers both evaluation paths: `static_assert` exercises the constexpr scalar
// path, and the runtime asserts exercise the SIMD path (NEON here).

#include <vkm/vkm.hpp>

#include <cmath>
#include <cstdio>

// A check that always runs, regardless of NDEBUG (unlike assert).
static int g_failures = 0;
#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

using namespace vkm;

// ---- compile-time (constexpr / scalar path) ----------------------------------

static_assert(float4{1, 2, 3, 4} + float4{10, 20, 30, 40} == float4{11, 22, 33, 44});
static_assert(float4{1, 2, 3, 4} * 2.0f == float4{2, 4, 6, 8});
static_assert(2.0f * float4{1, 2, 3, 4} == float4{2, 4, 6, 8});
static_assert(dot(float4{1, 2, 3, 4}, float4{1, 1, 1, 1}) == 10.0f);
static_assert(dot(float3{1, 0, 0}, float3{0, 1, 0}) == 0.0f);
static_assert(cross(float3{1, 0, 0}, float3{0, 1, 0}) == float3{0, 0, 1});
static_assert(float4{float3{1, 2, 3}, 4} == float4{1, 2, 3, 4});
static_assert(int3{1, 2, 3} * int3{2, 2, 2} == int3{2, 4, 6});

static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

int main() {
    // Runtime path hits the SIMD backend.
    float4 a{1, 2, 3, 4};
    float4 b{5, 6, 7, 8};

    CHECK((a + b) == float4(6, 8, 10, 12));
    CHECK((b - a) == float4(4, 4, 4, 4));
    CHECK((a * b) == float4(5, 12, 21, 32));
    CHECK((a * 2.0f) == float4(2, 4, 6, 8));
    CHECK(near(dot(a, b), 5 + 12 + 21 + 32));

    float3 n = normalize(float3{0, 3, 4});
    CHECK(near(length(n), 1.0f));
    CHECK(near(n.y, 3.0f / 5.0f) && near(n.z, 4.0f / 5.0f));

    CHECK(near(length(float4{0, 0, 0, 2}), 2.0f));

    if (g_failures == 0) {
        std::puts("vkm vector tests passed");
        return 0;
    }
    std::printf("vkm vector tests: %d failure(s)\n", g_failures);
    return 1;
}
