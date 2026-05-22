// vkm — Unity-style quaternion ops: angle / fromToRotation / lookRotation /
// rotateTowards / toQuat, plus invert/normalized and the direction constants.

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
    quat q90 = quat::fromAxisAngle(float3{0, 0, 1}, pi / 2);
    CHECK(near(angle(quat::identity(), q90), pi / 2));
    CHECK(near(angle(q90, q90), 0.0f));

    // ---- toQuat: inverse of toFloat3x3 (compare via rotation effect, sign-agnostic) ----
    for (quat q : {quat::fromAxisAngle(float3{0, 1, 0}, 0.7f),
                   quat::fromAxisAngle(float3{1, 2, 3}, 2.1f),
                   quat::fromAxisAngle(float3{-1, 0.5f, 2}, -1.3f)}) {
        quat rt = toQuat(toFloat3x3(q));
        for (float3 v : {float3{1, 0, 0}, float3{0, 1, 0}, float3{1, 2, 3}})
            CHECK(near(rotate(rt, v), rotate(q, v)));
    }

    // ---- fromToRotation ----
    CHECK(near(rotate(fromToRotation(float3{1, 0, 0}, float3{0, 1, 0}), float3{1, 0, 0}),
               float3{0, 1, 0}));
    CHECK(near(rotate(fromToRotation(float3{2, 0, 0}, float3{2, 0, 0}), float3{1, 0, 0}),
               float3{1, 0, 0})); // aligned -> identity
    CHECK(near(rotate(fromToRotation(float3{1, 0, 0}, float3{-1, 0, 0}), float3{1, 0, 0}),
               float3{-1, 0, 0})); // opposite -> 180 deg
    {
        float3 a{1, 2, 3}, b{-2, 1, 0.5f};
        CHECK(near(rotate(fromToRotation(a, b), normalize(a)), normalize(b)));
    }

    // ---- lookRotation: local forward (-Z) maps to the look direction, up stays up ----
    {
        quat q = lookRotation(forward, up); // already facing -Z, up +Y
        CHECK(near(rotate(q, forward), forward));
        CHECK(near(rotate(q, up), up));
        CHECK(near(q, quat::identity()));
    }
    {
        quat q = lookRotation(float3{1, 0, 0}); // face +X, default up
        CHECK(near(rotate(q, forward), float3{1, 0, 0})); // -Z -> +X
        CHECK(near(rotate(q, up), float3{0, 1, 0}));      // up preserved
    }
    for (float3 dir : {float3{0, 0, 1}, float3{1, 1, 0}, float3{1, 2, 2}}) {
        quat q = lookRotation(dir, up);
        CHECK(near(rotate(q, forward), normalize(dir))); // forward axis hits the target dir
    }

    // ---- rotateTowards(quat) ----
    {
        quat from = quat::identity(), to = q90;
        CHECK(near(angle(rotateTowards(from, to, pi), to), 0.0f));        // budget covers it -> to
        quat half = rotateTowards(from, to, pi / 4);                      // half the pi/2 arc
        CHECK(near(angle(from, half), pi / 4));
        CHECK(near(angle(half, to), pi / 4));
        CHECK(near(angle(rotateTowards(from, from, 1.0f), from), 0.0f));  // already there
    }

    // ---- inverse / invert / normalized ----
    {
        quat q = quat::fromAxisAngle(float3{0.2f, 1, -0.5f}, 1.1f);
        CHECK(near(rotate(q * q.inverse(), float3{1, 2, 3}), float3{1, 2, 3})); // q * q^-1 = id
        quat qi = q;
        qi.invert();
        CHECK(near(qi, q.inverse()));            // in-place matches the copy form
        CHECK(near(quat{1, 2, 3, 4}.normalized().len(), 1.0f)); // non-unit -> unit copy
    }

    if (g_failures == 0) {
        std::puts("vkm quat_ops tests passed");
        return 0;
    }
    std::printf("vkm quat_ops tests: %d failure(s)\n", g_failures);
    return 1;
}
