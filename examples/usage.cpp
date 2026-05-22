// vkm — sample usage. Build:
//   clang++ -std=c++23 -O2 -Iinclude examples/usage.cpp -o usage && ./usage
#include <vkm/vkm.hpp>

#include <cstdio>

using namespace vkm;

static void print(const char* label, float v) { std::printf("  %-26s %.4f\n", label, v); }
static void print(const char* label, float3 v) {
    std::printf("  %-26s (%.4f, %.4f, %.4f)\n", label, v.x, v.y, v.z);
}
static void print(const char* label, float4 v) {
    std::printf("  %-26s (%.4f, %.4f, %.4f, %.4f)\n", label, v.x, v.y, v.z, v.w);
}

int main() {
    // ---- 1. Adding float3 ----------------------------------------------------
    std::puts("[vectors]");
    float3 a{1, 2, 3};
    float3 b{4, 5, 6};
    print("a + b =", a + b);          // component-wise (SIMD for float4; scalar here)
    print("a * 2 =", a * 2.0f);       // vector * scalar

    // ---- 2. Cross / dot / magnitude -----------------------------------------
    print("cross(a, b) =", cross(a, b));
    print("dot(a, b) =", dot(a, b));
    print("length(a)            (mag)", length(a));
    print("length2(a)        (mag^2)", length2(a)); // no sqrt — cheaper for comparisons
    print("normalize(a) =", normalize(a));

    // ---- 3. Rotate a vector by a 30-degree quaternion about +Z --------------
    std::puts("\n[quaternion rotation]");
    quat q = quat::fromAxisAngle(float3{0, 0, 1}, rad(30.0f)); // axis, angle (radians)
    float3 v{1, 0, 0};
    print("rotate(q, v)   [fast path]", rotate(q, v));               // expect (cos30, sin30, 0)
    // The same rotation applied via the matrix form (w=0 -> direction, not point):
    float3 via_matrix = (toFloat4x4(q) * float4{v, 0.0f}).xyz();
    print("via toFloat4x4(q) * v", via_matrix);

    // ---- 4. MVP matrix multiplication ---------------------------------------
    std::puts("\n[MVP]");
    // model: scale 1, rotate q (30 deg / Z), translate to (0,1,0)  -> T * R * S
    float4x4 model = composeTRS(/*t*/ float3{0, 1, 0}, /*r*/ q, /*s*/ float3{1, 1, 1});
    float4x4 view = lookAt(/*eye*/ float3{0, 0, 5}, /*center*/ float3{0, 0, 0}, /*up*/ float3{0, 1, 0});
    float4x4 proj = perspective(rad(60.0f), 16.0f / 9.0f, 0.1f, 100.0f); // Vulkan clip

    float4x4 mvp = proj * view * model; // compose right-to-left, then M * v below

    float4 model_point{0, 0, 0, 1};         // a point in model space (w = 1)
    float4 clip = mvp * model_point;        // -> clip space
    float3 ndc = clip.xyz() * (1.0f / clip.w); // perspective divide -> NDC [0,1] depth
    print("clip = MVP * point", clip);
    print("ndc  = clip.xyz / clip.w", ndc);

    return 0;
}
