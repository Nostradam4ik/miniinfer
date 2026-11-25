#pragma once
// Converting a weight element to float inside (device) kernels. One templated interface for fp32/fp16/bf16.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename WT>
__device__ __forceinline__ float wt_to_float(WT v);

template <>
__device__ __forceinline__ float wt_to_float<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ float wt_to_float<__half>(__half v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float wt_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// Loads VEC adjacent weight elements at address p and converts them to float.
// For VEC > 1 the address must be aligned to VEC * sizeof(WT) (the GEMV launchers guarantee it:
// N is a multiple of VEC and pool buffers are aligned to 256 bytes).
template <typename WT, int VEC>
__device__ __forceinline__ void load_weights(const WT* p, float (&out)[VEC]) {
    if constexpr (VEC == 1) {
        out[0] = wt_to_float<WT>(p[0]);
    } else if constexpr (sizeof(WT) == 4) {
        if constexpr (VEC == 2) {
            const float2 v = *reinterpret_cast<const float2*>(p);
            out[0] = v.x;
            out[1] = v.y;
        } else {
            const float4 v = *reinterpret_cast<const float4*>(p);
            out[0] = v.x;
            out[1] = v.y;
            out[2] = v.z;
            out[3] = v.w;
        }
    } else {
        if constexpr (VEC == 2) {
            const unsigned int u = *reinterpret_cast<const unsigned int*>(p);
            const WT* e = reinterpret_cast<const WT*>(&u);
            out[0] = wt_to_float<WT>(e[0]);
            out[1] = wt_to_float<WT>(e[1]);
        } else {
            const uint2 u = *reinterpret_cast<const uint2*>(p);
            const WT* e = reinterpret_cast<const WT*>(&u);
            out[0] = wt_to_float<WT>(e[0]);
            out[1] = wt_to_float<WT>(e[1]);
            out[2] = wt_to_float<WT>(e[2]);
            out[3] = wt_to_float<WT>(e[3]);
        }
    }
}
