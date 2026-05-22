// vkm — determinant / inverse / normal-matrix tests.

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

static bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }
static bool near(float4 a, float4 b) {
    return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z) && near(a.w, b.w);
}
static bool near(const float3x3& a, const float3x3& b) {
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            if (!near(a.cols[c][r], b.cols[c][r])) return false;
    return true;
}
static bool near(const float4x4& a, const float4x4& b) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            if (!near(a.cols[c][r], b.cols[c][r])) return false;
    return true;
}

// ---- compile-time --------------------------------------------------------------

static_assert(determinant(float2x2::identity()) == 1.0f);
static_assert(determinant(float3x3::identity()) == 1.0f);
static_assert(determinant(float4x4::identity()) == 1.0f);
// Inverse of a translation is the negated translation (exact in float).
static_assert(inverse(translate(float3{1, 2, 3})) == translate(float3{-1, -2, -3}));

int main() {
    // determinant of a pure scale = product of the diagonal.
    CHECK(near(determinant(scale(float3{2, 3, 4})), 24.0f));

    // 2x2 / 3x3 inverse round-trips to identity.
    {
        float2x2 a{float2{4, 2}, float2{7, 6}};
        float2x2 prod = inverse(a) * a;
        CHECK(near(prod.cols[0][0], 1) && near(prod.cols[1][1], 1) &&
              near(prod.cols[0][1], 0) && near(prod.cols[1][0], 0));
    }

    float3x3 b{float3{1, 2, 3}, float3{0, 1, 4}, float3{5, 6, 0}};
    CHECK(near(inverse(b) * b, float3x3::identity()));

    // 4x4 inverse of a full TRS round-trips.
    float4x4 M = compose_trs(float3{3, -2, 5},
                             quat::from_axis_angle(float3{0.3f, 1, 0.5f}, radians(40)),
                             float3{2, 0.5f, 1.5f});
    CHECK(near(inverse(M) * M, float4x4::identity()));
    CHECK(near(M * inverse(M), float4x4::identity()));

    // inverse of a view matrix recovers a point (un-project the camera-space origin).
    float4x4 V = look_at(float3{1, 2, 3}, float3{0, 0, 0}, float3{0, 1, 0});
    CHECK(near(inverse(V) * float4{0, 0, 0, 1}, float4{1, 2, 3, 1})); // camera-space origin -> eye

    // normal matrix of a pure rotation equals that rotation's 3x3 (orthonormal).
    quat r = quat::from_axis_angle(float3{0, 1, 0}, radians(33));
    float4x4 R = to_float4x4(r);
    CHECK(near(normal_matrix(R), to_float3x3(r)));

    if (g_failures == 0) {
        std::puts("vkm inverse tests passed");
        return 0;
    }
    std::printf("vkm inverse tests: %d failure(s)\n", g_failures);
    return 1;
}
