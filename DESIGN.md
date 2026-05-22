# vkm — design

A modern (C++23), SIMD-first math library for Vulkan, with Slang/HLSL naming.
Header-only. Think "GLM, but Vulkan conventions by default and Slang-style
names so C++ and shader code mirror each other."

## Goals

- **Slang/HLSL naming.** `float3`, `float4x4`, `uint4` — not `vec3`/`mat4`.
- **SIMD from the start.** NEON / SSE / scalar backends behind one tiny seam.
- **Vulkan conventions baked in.** `[0,1]` depth + flipped Y for projections;
  `std140`/`std430` layout helpers for UBO/SSBO interop.
- **constexpr-friendly.** Everything usable at compile time; SIMD only at runtime.
- **Production-usable, but a learning vehicle.** Readable over clever.

## Layout

```
include/vkm/
  vkm.hpp            umbrella
  detail/simd.hpp    ISA backend (NEON / SSE / scalar). Only file with intrinsics.
  detail/layout.hpp  std140 / std430 alignment wrappers          [planned]
  vector.hpp         vector<T,N> + floatN/intN/uintN/boolN/doubleN/halfN
  matrix.hpp         matrix<T,R,C> + floatRxC                      [planned]
  quaternion.hpp     quat                                          [planned]
  transform.hpp      translate/rotate/scale, perspective (Vulkan)  [planned]
  common.hpp         clamp/lerp/min/max/abs/...                     [planned]
```

## Conventions

- **Names:** Slang/HLSL. Vectors `floatN`; matrices `floatRxC` (R rows, C cols);
  `uint` = `uint32_t`. Generic forms `vector<T,N>`, `matrix<T,R,C>` underneath.
- **`operator*` on vectors is component-wise** (matches Slang). Matrix product is
  `mul(a,b)` and `operator*`.
- **Storage:** components are named fields (`x,y,z,w`) — standard-layout, so
  `&v.x` is a contiguous array and the type drops straight into GPU buffers.
  `operator[]` uses a switch to stay valid in constant expressions.

### Matrix convention — DECISION PENDING

Recommended default: **column-major storage + column-vector math**
(`v' = M * v`, compose right-to-left `M = P * V * Model`).

- Matches Vulkan/SPIR-V's default column-major layout → upload a UBO matrix with
  a plain `memcpy`, no transpose.
- Matches GLM / textbook convention → tutorials and ported code just work.
- The Slang *name* (`float4x4`) is independent of CPU storage; a `mul()` mirroring
  shader semantics is provided alongside `operator*`.

If Slang is compiled with its row-major default, pass
`slangc -matrix-layout-column-major` or transpose on upload (will be documented).

## SIMD

`detail/simd.hpp` is the only file that includes intrinsics. It exposes a 4-wide
float register `f32x4` and a minimal op set (`load4/store4/splat/add/sub/mul/
div/neg/hsum`). Backends, chosen at compile time:

| Backend | When | Notes |
|---|---|---|
| NEON   | `__ARM_NEON` (Apple Silicon, ARM) | `vaddvq_f32` for horizontal sum |
| SSE    | x86-64 / SSE2+ | SSE3 `haddps` for horizontal sum |
| scalar | otherwise, or `-DVKM_FORCE_SCALAR` | always builds; SIMD/scalar diff testing |

Higher layers stay scalar-generic; only `float4` has SIMD overloads today. At
compile time (`if consteval`) those overloads use the scalar expressions, so the
whole API is constexpr. `float3` SIMD (padded load) and wider types come later.

## Roadmap

1. ✅ Project skeleton, SIMD backend, `vector<T,N>` (+ float4 SIMD), tests.
2. ⏳ `matrix<T,R,C>` once the matrix convention is confirmed.
3. `quat`, then `transform` (Vulkan-clip projections, look_at, TRS).
4. `common.hpp` (clamp/lerp/saturate/min/max/floor/...), swizzles.
5. `detail/layout.hpp`: `std140<T>` / `std430<T>` wrappers; static-assert struct
   offsets against the GLSL/Slang rules.
6. Wider SIMD (float3, double via AVX, half), benchmarks vs GLM.
```
