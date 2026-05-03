// MFMA real matmul test on MI300X (gfx942)
//
// Goal: verify the wave64 lane mapping for v_mfma_f32_16x16x16bf16_1k by
// computing a real (non-all-ones) 16x16 matrix multiply and comparing
// against a CPU reference.
//
// Test setup:
//   A[m,k] = (m + k * 0.125f) cast to bf16
//   B[k,n] = (k == n) ? 1.0 : 0.0   (identity)
//   Expected D = A * B = A  (identity multiplication)
//   So D[m,n] should equal A[m,n] = m + n * 0.125
//
// Lane mapping (AMDGPU CDNA3 wave64 16x16x16 bf16):
//   A operand: lane = (k/4) * 16 + m, slot within lane = k % 4
//              -> lane k holds A[k%16, (k/16)*4 .. (k/16)*4+3]
//   B operand: lane = (k/4) * 16 + n, slot within lane = k % 4
//              -> lane k holds B[(k/16)*4 .. (k/16)*4+3, k%16]
//   D output:  lane = (m/4) * 16 + n, slot within lane = m % 4
//              -> lane k holds D[(k/16)*4 .. (k/16)*4+3, k%16]
//
// We stage A,B in LDS (row-major), have each lane gather its 4 elements
// based on the mapping above, run MFMA, scatter the output back to LDS,
// then have lane 0 dump LDS to global memory for host verification.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>
#include <math.h>

typedef int16_t bf16x4 __attribute__((ext_vector_type(4)));
typedef float   f32x4  __attribute__((ext_vector_type(4)));

// Reinterpret a __hip_bfloat16 as int16 bit pattern
__device__ __forceinline__ int16_t bf16_bits(__hip_bfloat16 v) {
    int16_t out;
    __builtin_memcpy(&out, &v, sizeof(out));
    return out;
}

__global__ void mfma_matmul_kernel(const __hip_bfloat16* gA,
                                    const __hip_bfloat16* gB,
                                    float*                 gD) {
    __shared__ __hip_bfloat16 sA[16 * 16];
    __shared__ __hip_bfloat16 sB[16 * 16];
    __shared__ float           sD[16 * 16];

    int lane = threadIdx.x;

    // Load A and B from global to LDS (lanes 0..15 each load 16 elements)
    if (lane < 16) {
        for (int j = 0; j < 16; ++j) {
            sA[lane * 16 + j] = gA[lane * 16 + j];
            sB[lane * 16 + j] = gB[lane * 16 + j];
        }
    }
    __syncthreads();

    // Build per-lane bf16x4 operand A:
    // lane k holds A[k%16, (k/16)*4 + 0..3]
    int kblk = lane / 16;     // 0..3
    int row  = lane % 16;     // 0..15
    bf16x4 a;
    a[0] = bf16_bits(sA[row * 16 + kblk * 4 + 0]);
    a[1] = bf16_bits(sA[row * 16 + kblk * 4 + 1]);
    a[2] = bf16_bits(sA[row * 16 + kblk * 4 + 2]);
    a[3] = bf16_bits(sA[row * 16 + kblk * 4 + 3]);

    // Build per-lane bf16x4 operand B:
    // lane k holds B[(k/16)*4 + 0..3, k%16]
    int col = lane % 16;
    bf16x4 b;
    b[0] = bf16_bits(sB[(kblk * 4 + 0) * 16 + col]);
    b[1] = bf16_bits(sB[(kblk * 4 + 1) * 16 + col]);
    b[2] = bf16_bits(sB[(kblk * 4 + 2) * 16 + col]);
    b[3] = bf16_bits(sB[(kblk * 4 + 3) * 16 + col]);

    f32x4 c = {0.0f, 0.0f, 0.0f, 0.0f};
    f32x4 d = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);

    // Scatter output back to LDS:
    // lane k holds D[(k/16)*4 + 0..3, k%16]
    int mblk = lane / 16;     // 0..3
    int n    = lane % 16;     // 0..15
    sD[(mblk * 4 + 0) * 16 + n] = d[0];
    sD[(mblk * 4 + 1) * 16 + n] = d[1];
    sD[(mblk * 4 + 2) * 16 + n] = d[2];
    sD[(mblk * 4 + 3) * 16 + n] = d[3];

    __syncthreads();
    if (lane == 0) {
        for (int i = 0; i < 256; ++i) gD[i] = sD[i];
    }
}

int main() {
    __hip_bfloat16 hA[256], hB[256];
    float          hD_gpu[256];
    float          hD_ref[256];

    // A[m,n] = m + n * 0.125
    // B = identity
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            float v = float(m) + float(n) * 0.125f;
            hA[m * 16 + n] = __hip_bfloat16(v);
            hB[m * 16 + n] = __hip_bfloat16(m == n ? 1.0f : 0.0f);
        }
    }

    // CPU reference: D = A * B = A (since B is identity)
    // Exact (no bf16 rounding) — compute in fp32 from bf16 inputs
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < 16; ++k) {
                acc += float(hA[m * 16 + k]) * float(hB[k * 16 + n]);
            }
            hD_ref[m * 16 + n] = acc;
        }
    }

    __hip_bfloat16 *dA, *dB;
    float          *dD;
    hipMalloc(&dA, sizeof(hA));
    hipMalloc(&dB, sizeof(hB));
    hipMalloc(&dD, sizeof(hD_gpu));
    hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice);

    mfma_matmul_kernel<<<dim3(1), dim3(64)>>>(dA, dB, dD);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("[mfma-matmul] kernel error: %s\n", hipGetErrorString(err));
        return 1;
    }
    hipMemcpy(hD_gpu, dD, sizeof(hD_gpu), hipMemcpyDeviceToHost);

    int errors = 0;
    float max_abs_err = 0.0f;
    for (int i = 0; i < 256; ++i) {
        float diff = fabsf(hD_gpu[i] - hD_ref[i]);
        if (diff > max_abs_err) max_abs_err = diff;
        if (diff > 0.01f) {
            if (errors < 4) {
                int m = i / 16, n = i % 16;
                printf("[mfma-matmul] mismatch at D[%d,%d]: gpu=%.4f ref=%.4f\n",
                       m, n, hD_gpu[i], hD_ref[i]);
            }
            ++errors;
        }
    }
    printf("[mfma-matmul] errors=%d/256 max_abs_err=%.4f  %s\n",
           errors, max_abs_err, errors == 0 ? "PASS" : "FAIL");
    printf("[mfma-matmul] D[0,0..3]=%.3f %.3f %.3f %.3f  D[1,0..3]=%.3f %.3f %.3f %.3f\n",
           hD_gpu[0], hD_gpu[1], hD_gpu[2], hD_gpu[3],
           hD_gpu[16], hD_gpu[17], hD_gpu[18], hD_gpu[19]);
    printf("[mfma-matmul] ref D[0,0..3]=%.3f %.3f %.3f %.3f  ref D[1,0..3]=%.3f %.3f %.3f %.3f\n",
           hD_ref[0], hD_ref[1], hD_ref[2], hD_ref[3],
           hD_ref[16], hD_ref[17], hD_ref[18], hD_ref[19]);

    hipFree(dA); hipFree(dB); hipFree(dD);
    return errors == 0 ? 0 : 1;
}
