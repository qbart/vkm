// vkm — GPU packing tests: snorm/unorm quantization round trips + octahedral
// normal/tangent encode/decode.

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

static bool nearf(float a, float b, float eps) { return std::fabs(a - b) < eps; }

// A couple of constexpr round trips (the offered static_assert form).
static_assert(dequantize_unorm16(quantize_unorm16(1.0f)) == 1.0f);
static_assert(quantize_snorm10(1.0f) == 511 && quantize_unorm10(1.0f) == 1023);

int main() {
    // ---- snorm round trips: error <= 0.5 / scale ----
    for (float v : {-1.0f, -0.5f, -0.123f, 0.0f, 0.25f, 0.5f, 1.0f}) {
        CHECK(nearf(dequantize_snorm16(quantize_snorm16(v)), v, 4e-5f));
        CHECK(nearf(dequantize_snorm8(quantize_snorm8(v)), v, 1e-2f));
        CHECK(nearf(dequantize_snorm10(quantize_snorm10(v)), v, 2.5e-3f));
    }
    // ---- unorm round trips ----
    for (float v : {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f}) {
        CHECK(nearf(dequantize_unorm16(quantize_unorm16(v)), v, 2e-5f));
        CHECK(nearf(dequantize_unorm8(quantize_unorm8(v)), v, 4e-3f));
        CHECK(nearf(dequantize_unorm10(quantize_unorm10(v)), v, 1e-3f));
    }
    // ---- clamping: encode clamps the input, snorm decode clamps the output ----
    CHECK(quantize_snorm16(2.0f) == 32767 && quantize_snorm16(-2.0f) == -32767);
    CHECK(nearf(dequantize_snorm16(-32768), -1.0f, 1e-6f)); // most-negative code -> -1
    CHECK(quantize_unorm8(2.0f) == 255 && quantize_unorm8(-1.0f) == 0);

    // ---- octahedral normal round trip (incl. both hemispheres + axes/poles) ----
    for (float3 d : {float3{1, 0, 0}, float3{0, 1, 0}, float3{0, 0, 1}, float3{0, 0, -1},
                     float3{-1, 0, 0}, float3{1, 1, 1}, float3{-1, 2, -3},
                     float3{0.3f, -0.7f, 0.6f}, float3{2, -1, -2}}) {
        float3 n = normalize(d);
        float3 r = oct_decode16(oct_encode16(n));
        CHECK(nearf(length(r), 1.0f, 1e-4f)); // unproject re-normalizes
        CHECK(dot(n, r) > 0.999f);             // 16-bit oct is accurate
    }

    // ---- tangent packing: handedness in the LSB, direction preserved ----
    float3 t = normalize(float3{1, 2, 3});
    for (float h : {1.0f, -1.0f}) {
        short2 enc = oct_encode_tangent(t, h);
        CHECK((enc.x & 1) == (h < 0.0f ? 1 : 0)); // LSB encodes the sign
        float dh = 0.0f;
        float3 dt = oct_decode_tangent(enc, dh);
        CHECK(dh == h);
        CHECK(dot(t, dt) > 0.999f);
        CHECK(nearf(length(dt), 1.0f, 1e-4f));
    }
    // bitangent reconstruction (caller-side), as documented:
    {
        float3 n = normalize(float3{0, 1, 0});
        short2 enc = oct_encode_tangent(t, -1.0f);
        float dh = 0.0f;
        float3 dt = oct_decode_tangent(enc, dh);
        float3 bitangent = cross(n, dt) * dh;
        CHECK(nearf(length(bitangent), length(cross(n, dt)), 1e-4f)); // sign-flip keeps magnitude
    }

    if (g_failures == 0) {
        std::puts("vkm pack tests passed");
        return 0;
    }
    std::printf("vkm pack tests: %d failure(s)\n", g_failures);
    return 1;
}
