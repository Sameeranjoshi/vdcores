#include "hip/hip_runtime.h"
#pragma once

#include <cstdint>
#include "dae/hip_compat.cuh"

#ifndef __HIP_PLATFORM_AMD__
#include <cutlass/numeric_types.h>
#endif

template<typename T> struct F16Traits;

template<> struct F16Traits<half> {
    using vec2_t = half2;
    static __device__ __forceinline__ float2 to_float2(half2 v)    { return __half22float2(v); }
    static __device__ __forceinline__ half2  from_float2(float2 v) { return __float22half2_rn(v); }
    static __device__ __forceinline__ float  to_float(half e)    { return __half2float(e); }
};

#ifndef __HIP_PLATFORM_AMD__

template<> struct F16Traits<__hip_bfloat16> {
    using vec2_t = __hip_bfloat162;
    static __device__ __forceinline__ float2        to_float2(__hip_bfloat162 v) { return __bfloat1622float2(v); }
    static __device__ __forceinline__ __hip_bfloat162 from_float2(float2 v)      { return __float22bfloat162_rn(v); }
    static __device__ __forceinline__ float          to_float(__hip_bfloat16 e){ return __bfloat162float(e); }
};

template<>
struct F16Traits<cutlass::bfloat16_t> : F16Traits<__hip_bfloat16> {
    static __device__ __forceinline__ float to_float(cutlass::bfloat16_t e) {
        return F16Traits<__hip_bfloat16>::to_float(static_cast<__hip_bfloat16>(e));
    }
};

#else  // __HIP_PLATFORM_AMD__

// AMD bf16 path: ROCm 7.x exposes __hip_bfloat16 / __hip_bfloat162 (full
// structs with .x / .y, matching the CUDA naming convention) through
// <hip/hip_bf16.h>. We rely on those — see hip_compat.cuh for the include —
// and provide a scalar-only F16Traits. Fast vector bf16 instructions exist on
// CDNA but aren't wired up here; the f32<->bf16 conversions used by RMSNorm /
// argmax go through 32-bit lanes.
template<> struct F16Traits<__hip_bfloat16> {
    using vec2_t = __hip_bfloat162;
    static __device__ __forceinline__ float2 to_float2(__hip_bfloat162 v) {
        return float2{ float(v.x), float(v.y) };
    }
    static __device__ __forceinline__ __hip_bfloat162 from_float2(float2 v) {
        __hip_bfloat162 r;
        r.x = __hip_bfloat16(v.x);
        r.y = __hip_bfloat16(v.y);
        return r;
    }
    static __device__ __forceinline__ float to_float(__hip_bfloat16 e) {
        return float(e);
    }
};

#endif  // __HIP_PLATFORM_AMD__
