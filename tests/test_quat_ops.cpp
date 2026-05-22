// vkm — Unity-style quaternion ops: angle / from_to_rotation / look_rotation /
// rotate_towards / to_quat, plus Invert/Normalized and the direction constants.

#include <vkm/vkm.hpp>

#include <cmath>
#include <cstdio>
#include <initializer_list>

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
static bool near(quat a, quat b) {
    return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z) && near(a.w, b.w);
}

int main() {
    // ---- direction constants: right-handed, Y-up, forward = -Z ----
    CHECK(near(up, float3{0, 1, 0}));
    CHECK(near(forward, float3{0, 0, -1}));
    CHECK(near(forward, -back));         // forward == -back
    CHECK(near(cross(right, up), back)); // RH basis: X × Y = +Z = back

    // ---- angle(quat, quat), radians ----
    quat q90 = quat::from_axis_angle(float3{0, 0, 1}, pi / 2);
    CHECK(near(angle(quat::identity(), q90), pi / 2));
    CHECK(near(angle(q90, q90), 0.0f));

    // ---- to_quat: inverse of to_float3x3 (compare via rotation effect, sign-agnostic) ----
    for (quat q : {quat::from_axis_angle(float3{0, 1, 0}, 0.7f),
                   quat::from_axis_angle(float3{1, 2, 3}, 2.1f),
                   quat::from_axis_angle(float3{-1, 0.5f, 2}, -1.3f)}) {
        quat rt = to_quat(to_float3x3(q));
        for (float3 v : {float3{1, 0, 0}, float3{0, 1, 0}, float3{1, 2, 3}})
            CHECK(near(rotate(rt, v), rotate(q, v)));
    }

    // ---- from_to_rotation ----
    CHECK(near(rotate(from_to_rotation(float3{1, 0, 0}, float3{0, 1, 0}), float3{1, 0, 0}),
               float3{0, 1, 0}));
    CHECK(near(rotate(from_to_rotation(float3{2, 0, 0}, float3{2, 0, 0}), float3{1, 0, 0}),
               float3{1, 0, 0})); // aligned -> identity
    CHECK(near(rotate(from_to_rotation(float3{1, 0, 0}, float3{-1, 0, 0}), float3{1, 0, 0}),
               float3{-1, 0, 0})); // opposite -> 180 deg
    {
        float3 a{1, 2, 3}, b{-2, 1, 0.5f};
        CHECK(near(rotate(from_to_rotation(a, b), normalize(a)), normalize(b)));
    }

    // ---- look_rotation: local forward (-Z) maps to the look direction, up stays up ----
    {
        quat q = look_rotation(forward, up); // already facing -Z, up +Y
        CHECK(near(rotate(q, forward), forward));
        CHECK(near(rotate(q, up), up));
        CHECK(near(q, quat::identity()));
    }
    {
        quat q = look_rotation(float3{1, 0, 0}); // face +X, default up
        CHECK(near(rotate(q, forward), float3{1, 0, 0})); // -Z -> +X
        CHECK(near(rotate(q, up), float3{0, 1, 0}));      // up preserved
    }
    for (float3 dir : {float3{0, 0, 1}, float3{1, 1, 0}, float3{1, 2, 2}}) {
        quat q = look_rotation(dir, up);
        CHECK(near(rotate(q, forward), normalize(dir))); // forward axis hits the target dir
    }

    // ---- rotate_towards(quat) ----
    {
        quat from = quat::identity(), to = q90;
        CHECK(near(angle(rotate_towards(from, to, pi), to), 0.0f));        // budget covers it -> to
        quat half = rotate_towards(from, to, pi / 4);                      // half the pi/2 arc
        CHECK(near(angle(from, half), pi / 4));
        CHECK(near(angle(half, to), pi / 4));
        CHECK(near(angle(rotate_towards(from, from, 1.0f), from), 0.0f));  // already there
    }

    // ---- Inverse / Invert / Normalized ----
    {
        quat q = quat::from_axis_angle(float3{0.2f, 1, -0.5f}, 1.1f);
        CHECK(near(rotate(q * q.Inverse(), float3{1, 2, 3}), float3{1, 2, 3})); // q * q^-1 = id
        quat qi = q;
        qi.Invert();
        CHECK(near(qi, q.Inverse()));            // in-place matches the copy form
        CHECK(near(quat{1, 2, 3, 4}.Normalized().Len(), 1.0f)); // non-unit -> unit copy
    }

    if (g_failures == 0) {
        std::puts("vkm quat_ops tests passed");
        return 0;
    }
    std::printf("vkm quat_ops tests: %d failure(s)\n", g_failures);
    return 1;
}
