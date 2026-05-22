// vkm — transform + quaternion + Vulkan-projection tests.

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
static bool near(float4 a, float4 b) {
    return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z) && near(a.w, b.w);
}

int main() {
    // --- quaternion: 90 deg about +Z maps +X -> +Y ---
    quat qz = quat::from_axis_angle(float3{0, 0, 1}, radians(90));
    CHECK(near(rotate(qz, float3{1, 0, 0}), float3{0, 1, 0}));         // optimized rotate
    CHECK(near(to_float4x4(qz) * float4{1, 0, 0, 1}, float4{0, 1, 0, 1})); // via matrix
    // Composition order: q*q = 180 deg about Z maps +X -> -X.
    CHECK(near(rotate(qz * qz, float3{1, 0, 0}), float3{-1, 0, 0}));

    // --- quaternion interpolation ---
    // halfway between identity and a 90deg-about-Z rotation is a 45deg rotation.
    quat half = slerp(quat::identity(), qz, 0.5f);
    float s = 0.70710678f; // sin(45) = cos(45)
    CHECK(near(rotate(half, float3{1, 0, 0}), float3{s, s, 0}));
    // endpoints are exact.
    CHECK(near(rotate(slerp(quat::identity(), qz, 0.0f), float3{1, 0, 0}), float3{1, 0, 0}));
    CHECK(near(rotate(slerp(quat::identity(), qz, 1.0f), float3{1, 0, 0}), float3{0, 1, 0}));
    // nlerp lands near the same place at the midpoint (close orientations).
    CHECK(near(rotate(nlerp(quat::identity(), qz, 0.5f), float3{1, 0, 0}), float3{s, s, 0}));

    // --- affine ---
    CHECK(near(translate(float3{10, 20, 30}) * float4{1, 2, 3, 1}, float4{11, 22, 33, 1}));
    CHECK(near(scale(float3{2, 3, 4}) * float4{1, 1, 1, 1}, float4{2, 3, 4, 1}));

    // --- look_at (RH): camera 5 units down +Z, looking at origin ---
    float4x4 V = look_at(float3{0, 0, 5}, float3{0, 0, 0}, float3{0, 1, 0});
    CHECK(near(V * float4{0, 0, 0, 1}, float4{0, 0, -5, 1})); // origin -> 5 in front (-Z)
    CHECK(near(V * float4{0, 0, 5, 1}, float4{0, 0, 0, 1}));  // eye -> view origin

    // --- perspective: Vulkan depth [0,1] ---
    float n = 0.1f, f = 100.0f;
    float4x4 P = perspective(radians(60), 16.0f / 9.0f, n, f);
    float4 cnear = P * float4{0, 0, -n, 1}; // point on near plane
    float4 cfar = P * float4{0, 0, -f, 1};  // point on far plane
    CHECK(near(cnear.z / cnear.w, 0.0f));   // near -> 0
    CHECK(near(cfar.z / cfar.w, 1.0f));     // far  -> 1

    // --- perspective: Y is flipped (a point ABOVE center -> negative clip Y) ---
    float4 cup = P * float4{0, 1, -1, 1};
    CHECK(cup.y < 0.0f);

    // --- ortho: Vulkan depth [0,1] ---
    float4x4 O = ortho(-1, 1, -1, 1, n, f);
    CHECK(near((O * float4{0, 0, -n, 1}).z, 0.0f));
    CHECK(near((O * float4{0, 0, -f, 1}).z, 1.0f));

    // --- a full MVP pipeline: model point through TRS -> view -> proj ---
    float4x4 M = compose_trs(float3{0, 0, 0}, quat::identity(), float3{1, 1, 1});
    float4x4 mvp = P * V * M;
    float4 clip = mvp * float4{0, 0, 0, 1}; // origin should be in front of camera
    CHECK(clip.w > 0.0f);                   // positive w -> in front, will divide sanely

    if (g_failures == 0) {
        std::puts("vkm transform tests passed");
        return 0;
    }
    std::printf("vkm transform tests: %d failure(s)\n", g_failures);
    return 1;
}
