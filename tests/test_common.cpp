// vkm — swizzle + common-function tests.

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

static bool near(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) < eps; }

// ---- swizzles (constexpr) -----------------------------------------------------

static_assert(float4{1, 2, 3, 4}.xyz() == float3{1, 2, 3});       // drop w
static_assert(float4{1, 2, 3, 4}.xy() == float2{1, 2});
static_assert(float4{1, 2, 3, 4}.zw() == float2{3, 4});
static_assert(float3{1, 2, 3}.yz() == float2{2, 3});
static_assert(float2{1, 2}.yx() == float2{2, 1});
static_assert(float4{1, 2, 3, 4}.swizzle<3, 2, 1, 0>() == float4{4, 3, 2, 1}); // reverse
static_assert(swizzle<0, 0, 0>(float4{5, 6, 7, 8}) == float3{5, 5, 5});        // splat
static_assert(float4{1, 2, 3, 4}.rgb() == float3{1, 2, 3});

// ---- common (constexpr where possible) ----------------------------------------

static_assert(clamp(5.0f, 0.0f, 1.0f) == 1.0f);
static_assert(saturate(-3.0f) == 0.0f);
static_assert(lerp(0.0f, 10.0f, 0.25f) == 2.5f);
static_assert(lerp(float3{0, 0, 0}, float3{2, 4, 6}, 0.5f) == float3{1, 2, 3});
static_assert(min(float3{1, 5, 3}, float3{4, 2, 6}) == float3{1, 2, 3});
static_assert(max(float3{1, 5, 3}, 4.0f) == float3{4, 5, 4});
static_assert(abs(float3{-1, 2, -3}) == float3{1, 2, 3});
static_assert(step(0.5f, 0.75f) == 1.0f);
static_assert(reflect(float3{1, -1, 0}, float3{0, 1, 0}) == float3{1, 1, 0}); // bounce off floor
static_assert(distance2(float3{0, 0, 0}, float3{0, 3, 4}) == 25.0f);          // 5^2, no sqrt

int main() {
    // Runtime cmath-backed functions.
    CHECK(near(rsqrt(4.0f), 0.5f));
    CHECK(near(frac(3.25f), 0.25f));
    CHECK(near(distance(float3{0, 0, 0}, float3{0, 3, 4}), 5.0f));

    float3 v = sqrt(float3{4, 9, 16});
    CHECK(near(v.x, 2) && near(v.y, 3) && near(v.z, 4));

    float3 f = floor(float3{1.7f, 2.2f, -0.3f});
    CHECK(near(f.x, 1) && near(f.y, 2) && near(f.z, -1));

    CHECK(near(smoothstep(0.0f, 1.0f, 0.5f), 0.5f));

    if (g_failures == 0) {
        std::puts("vkm common tests passed");
        return 0;
    }
    std::printf("vkm common tests: %d failure(s)\n", g_failures);
    return 1;
}
