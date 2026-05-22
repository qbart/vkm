// vkm — Unity-style vector ops: angle / project / project_on_plane /
// move_towards / rotate_towards (plus a reflect sanity check).

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

int main() {
    // ---- angle (radians, [0, pi]); inputs need not be unit ----
    CHECK(near(angle(float3{1, 0, 0}, float3{0, 1, 0}), pi / 2));   // perpendicular
    CHECK(near(angle(float3{2, 0, 0}, float3{5, 0, 0}), 0.0f));     // parallel
    CHECK(near(angle(float3{1, 0, 0}, float3{-1, 0, 0}), pi));      // opposite
    CHECK(near(angle(float3{3, 0, 0}, float3{0, 0.5f, 0}), pi / 2)); // non-unit, still pi/2
    CHECK(near(angle(float3{0, 0, 0}, float3{1, 0, 0}), 0.0f));     // degenerate -> 0, no NaN

    // ---- project / project_on_plane ----
    CHECK(near(project(float3{2, 3, 0}, float3{1, 0, 0}), float3{2, 0, 0}));   // onto +X
    CHECK(near(project(float3{1, 5, 1}, float3{0, 2, 0}), float3{0, 5, 0}));   // non-unit normal
    CHECK(near(project(float3{1, 2, 3}, float3{0, 0, 0}), float3{0, 0, 0}));   // zero normal -> 0
    CHECK(near(project_on_plane(float3{1, 2, 3}, float3{0, 0, 1}), float3{1, 2, 0})); // drop Z
    CHECK(near(project_on_plane(float3{1, 2, 3}, float3{0, 0, 0}), float3{1, 2, 3})); // zero -> v
    // identity: projection + in-plane component reconstruct the original vector
    {
        float3 v{1, 2, 3}, n{0, 1, 0};
        CHECK(near(project(v, n) + project_on_plane(v, n), v));
    }

    // ---- reflect (already in common.hpp; sanity that it matches Unity's form) ----
    CHECK(near(reflect(float3{1, -1, 0}, float3{0, 1, 0}), float3{1, 1, 0}));

    // ---- move_towards (scalar + vector) ----
    CHECK(near(move_towards(0.0f, 10.0f, 3.0f), 3.0f));   // step
    CHECK(near(move_towards(0.0f, 10.0f, 100.0f), 10.0f)); // clamp, no overshoot
    CHECK(near(move_towards(10.0f, 0.0f, 3.0f), 7.0f));   // toward smaller
    CHECK(near(move_towards(float3{0, 0, 0}, float3{10, 0, 0}, 3.0f), float3{3, 0, 0}));
    CHECK(near(move_towards(float3{0, 0, 0}, float3{10, 0, 0}, 100.0f), float3{10, 0, 0})); // clamp

    // ---- rotate_towards ----
    // enough angular budget -> snap onto target direction (lengths equal -> len stays 1)
    CHECK(near(rotate_towards(float3{1, 0, 0}, float3{0, 1, 0}, pi, 0.0f), float3{0, 1, 0}));
    // partial 45-degree turn within the plane, unit length preserved
    CHECK(near(rotate_towards(float3{1, 0, 0}, float3{0, 1, 0}, pi / 4, 0.0f),
               float3{std::cos(pi / 4), std::sin(pi / 4), 0}));
    // magnitude steps toward |target| by at most max_magnitude_delta (dir snaps via large angle budget)
    CHECK(near(rotate_towards(float3{1, 0, 0}, float3{0, 5, 0}, pi, 2.0f), float3{0, 3, 0}));
    // antiparallel with full budget -> lands on target (the degenerate-axis path)
    CHECK(near(rotate_towards(float3{1, 0, 0}, float3{-1, 0, 0}, pi, 0.0f), float3{-1, 0, 0}));
    // antiparallel, partial turn -> stays unit length and is perpendicular to the start
    {
        float3 r = rotate_towards(float3{1, 0, 0}, float3{-1, 0, 0}, pi / 2, 0.0f);
        CHECK(near(length(r), 1.0f));
        CHECK(near(dot(r, float3{1, 0, 0}), 0.0f));
    }
    // zero start: no direction, take target's direction at the stepped magnitude
    CHECK(near(rotate_towards(float3{0, 0, 0}, float3{0, 3, 0}, pi, 2.0f), float3{0, 2, 0}));

    if (g_failures == 0) {
        std::puts("vkm vector_ops tests passed");
        return 0;
    }
    std::printf("vkm vector_ops tests: %d failure(s)\n", g_failures);
    return 1;
}
