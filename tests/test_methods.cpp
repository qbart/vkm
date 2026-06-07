// vkm — member-style API + compound operators.

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
static bool near(float3 a, float3 b) { return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z); }
static bool near(const float4x4& a, const float4x4& b) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            if (!near(a.cols[c][r], b.cols[c][r])) return false;
    return true;
}

// ---- compile-time: vector members + compound ops are constexpr ---------------

static_assert(float3{0, 3, 4}.length2() == 25.0f);
static_assert(float3{-1, 2, -3}.abs() == float3{1, 2, 3});
static_assert(float3{5, -1, 0.5f}.saturated() == float3{1, 0, 0.5f});
static_assert(float3{5, -1, 3}.clamped(0.0f, 2.0f) == float3{2, 0, 2});
static_assert(float3{1, -1, 0}.reflected(float3{0, 1, 0}) == float3{1, 1, 0});
static_assert([] { float3 v{1, 2, 3}; v += float3{1, 1, 1}; return v; }() == float3{2, 3, 4});
static_assert([] { float4 v{1, 2, 3, 4}; v *= 2.0f; return v; }() == float4{2, 4, 6, 8});
static_assert(quat{1, 2, 3, 4}.conjugate() == quat{-1, -2, -3, 4});

int main() {
    // ---- vector ----
    float3 v{0, 3, 4};
    CHECK(near(v.length(), 5.0f));
    CHECK(near(v.normalized(), float3{0, 0.6f, 0.8f}));
    CHECK(near(v, float3{0, 3, 4})); // normalized() did not mutate

    float3 n{0, 0, 2};
    n.normalize(); // mutates
    CHECK(near(n, float3{0, 0, 1}));

    float3 p{1, 2, 3};
    CHECK(p.ptr()[0] == 1 && p.ptr()[1] == 2 && p.ptr()[2] == 3); // contiguous

    float4 a{1, 2, 3, 4};
    a += float4{10, 10, 10, 10};
    CHECK((a == float4{11, 12, 13, 14}));
    a -= float4{1, 2, 3, 4};
    CHECK((a == float4{10, 10, 10, 10}));
    a *= 0.5f;
    CHECK((a == float4{5, 5, 5, 5}));
    float4 b{2, 2, 2, 2};
    b *= float4{3, 3, 3, 3};
    CHECK((b == float4{6, 6, 6, 6}));

    // ---- quat ----
    quat q = quat::from_axis_angle(float3{0, 0, 1}, rad(90));
    CHECK(near(q.length(), 1.0f));
    // q * inverse(q) is the identity rotation.
    CHECK(near(rotate(q * q.inverse(), float3{1, 0, 0}), float3{1, 0, 0}));
    // For a unit quaternion, conjugate() == inverse().
    CHECK(near(rotate(q.conjugate(), float3{0, 1, 0}), rotate(q.inverse(), float3{0, 1, 0})));
    CHECK(q.ptr()[3] == q.w);
    quat acc = quat::identity();
    acc *= q; // in-place Hamilton product
    CHECK(near(rotate(acc, float3{1, 0, 0}), rotate(q, float3{1, 0, 0})));

    // ---- matrix ----
    float4x4 M = compose_trs(float3{1, 2, 3}, q, float3{2, 2, 2});
    CHECK(near(M.determinant(), determinant(M)));
    CHECK(near(M * M.inverse(), float4x4::identity()));
    CHECK(M.transposed().transposed() == M); // double transpose round-trips

    float4x4 mt = M;
    mt.transpose(); // in place
    CHECK(mt == M.transposed());

    float4x4 mi = M;
    mi.invert(); // in place
    CHECK(near(mi, M.inverse()));

    // ptr is column-major: a translation's components land in the last 4 floats.
    float4x4 T = translate(float3{7, 8, 9});
    CHECK(T.ptr()[12] == 7 && T.ptr()[13] == 8 && T.ptr()[14] == 9 && T.ptr()[15] == 1);

    float4x4 I2 = float4x4::identity();
    I2 += float4x4::identity();
    CHECK(near(I2, 2.0f * float4x4::identity()));

    if (g_failures == 0) {
        std::puts("vkm methods tests passed");
        return 0;
    }
    std::printf("vkm methods tests: %d failure(s)\n", g_failures);
    return 1;
}
