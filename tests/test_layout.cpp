// vkm — std140/std430 layout tests.
//
// Every assert below is a number straight from the GLSL/Vulkan layout rules.
// If one breaks, the encoded rule is wrong.

#include <vkm/vkm.hpp>

#include <cstddef>
#include <cstdio>

using namespace vkm;

// ---- base alignment traits ----------------------------------------------------

static_assert(std140_align_v<float> == 4);
static_assert(std140_align_v<float2> == 8);
static_assert(std140_align_v<float3> == 16); // the classic gotcha
static_assert(std140_align_v<float4> == 16);
static_assert(std140_align_v<double2> == 16);
static_assert(std140_align_v<double3> == 32);
static_assert(std430_align_v<float3> == 16); // vec3 is 16 in std430 too
static_assert(std430_align_v<float2x2> == 8); // column = vec2 in std430

// ---- scalar / vector wrappers -------------------------------------------------

static_assert(sizeof(std140<float>) == 4 && alignof(std140<float>) == 4);
static_assert(sizeof(std140<float2>) == 8 && alignof(std140<float2>) == 8);
static_assert(sizeof(std140<float3>) == 16 && alignof(std140<float3>) == 16);
static_assert(sizeof(std140<float4>) == 16 && alignof(std140<float4>) == 16);

// ---- matrix wrappers (columns padded to slots) --------------------------------

static_assert(sizeof(std140<float4x4>) == 64 && alignof(std140<float4x4>) == 16);
static_assert(sizeof(std140<float3x3>) == 48 && alignof(std140<float3x3>) == 16); // 3 cols * 16
static_assert(sizeof(std140<float2x2>) == 32 && alignof(std140<float2x2>) == 16); // 2 cols * 16
static_assert(sizeof(std430<float4x4>) == 64);
static_assert(sizeof(std430<float3x3>) == 48); // vec3 columns -> 16 each
static_assert(sizeof(std430<float2x2>) == 16); // vec2 columns -> 8 each (no round)

// ---- arrays -------------------------------------------------------------------

static_assert(sizeof(std140_array<float, 4>) == 64); // stride 16 each
static_assert(sizeof(std140_array<float4, 2>) == 32);
static_assert(sizeof(std430_array<float, 4>) == 16); // stride 4 each
static_assert(sizeof(std430_array<float2, 3>) == 24); // stride 8 each
static_assert(sizeof(std430_array<float3, 2>) == 32); // vec3 stride still 16

// ---- a real UBO block, wrapper path -------------------------------------------
// Matches a Slang/GLSL std140 block declared field-for-field the same way
// (note: uses float4 for the eye position, sidestepping the vec3+scalar tuck).
struct CameraData {
    std140<float4x4> view;   // offset 0
    std140<float4x4> proj;   // offset 64
    std140<float4> eye;      // offset 128
    std140<float2> jitter;   // offset 144
    std140<float> near_;     // offset 152
    std140<float> far_;      // offset 156
};
static_assert(offsetof(CameraData, view) == 0);
static_assert(offsetof(CameraData, proj) == 64);
static_assert(offsetof(CameraData, eye) == 128);
static_assert(offsetof(CameraData, jitter) == 144);
static_assert(offsetof(CameraData, near_) == 152);
static_assert(offsetof(CameraData, far_) == 156);
static_assert(sizeof(CameraData) == 160 && alignof(CameraData) == 16);

// ---- byte-exact "tuck-in", alignas-trait path ---------------------------------
// A Slang block { vec3 c; float d; } packs d into the vec3's 4th lane (offset
// 12). The wrapper path can't express that; the alignas-trait path can.
struct alignas(16) Tuck {
    alignas(std140_align_v<float3>) float3 c; // align 16, natural size 12
    float d;                                  // tucks in at offset 12
};
static_assert(offsetof(Tuck, c) == 0);
static_assert(offsetof(Tuck, d) == 12); // <-- the tuck-in
static_assert(sizeof(Tuck) == 16);

int main() {
    // Round-trip a matrix through the std140 wrapper (memcpy-able to a UBO).
    float4x4 m = float4x4{float4{1, 0, 0, 0}, float4{0, 1, 0, 0},
                          float4{0, 0, 1, 0}, float4{7, 8, 9, 1}};
    std140<float4x4> packed = m;
    float4x4 back = packed;
    bool ok = (back == m);

    std140_array<float4, 3> arr;
    arr[1] = float4{1, 2, 3, 4};
    float4 read = arr[1];
    ok = ok && (read == float4{1, 2, 3, 4});

    if (ok) {
        std::puts("vkm layout tests passed");
        return 0;
    }
    std::puts("vkm layout tests FAILED");
    return 1;
}
