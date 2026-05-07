// MFMA smoke test on MI300X (gfx942)
//
// Goal: prove the bf16 16x16x16 MFMA intrinsic works on this toolchain
// before integrating into the dae2 interpreter for GEMV/GEMM.
//
// Test: A = 16x16 of bf16(1.0), B = 16x16 of bf16(1.0), C = 16x16 of fp32(0).
// Expected D = A*B+C = 16x16 of fp32(16.0) (each element = sum over K=16 of 1*1).
//
// Wave64 layout: each of 64 lanes holds 4 A elements, 4 B elements,
// accumulates 4 C/D elements. We don't care about the exact lane->matrix
// mapping for this test — every output should be 16.0 regardless.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdio.h>

// AMDGPU MFMA intrinsic types: clang ext_vector_type
typedef int16_t bf16x4 __attribute__((ext_vector_type(4)));
typedef float   f32x4  __attribute__((ext_vector_type(4)));

__global__ void mfma_smoke_kernel(float* out) {
    // bf16(1.0) bit pattern = top 16 bits of fp32(1.0)=0x3F800000 -> 0x3F80
    constexpr int16_t BF16_ONE = 0x3F80;
    bf16x4 a = {BF16_ONE, BF16_ONE, BF16_ONE, BF16_ONE};
    bf16x4 b = {BF16_ONE, BF16_ONE, BF16_ONE, BF16_ONE};
    f32x4 c = {0.0f, 0.0f, 0.0f, 0.0f};

    // 16x16x16 bf16 MFMA: D = A*B + C
    // For 16x16x16 with M=N=K=16, each output element = sum_{k=0..15} A[m,k]*B[k,n]
    // With all-ones inputs and zero accumulator, every output = 16.0
    f32x4 d = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);

    int lane = threadIdx.x;
    out[lane * 4 + 0] = d[0];
    out[lane * 4 + 1] = d[1];
    out[lane * 4 + 2] = d[2];
    out[lane * 4 + 3] = d[3];
}

int main() {
    const int N_OUT = 64 * 4;  // 64 lanes × 4 fp32 = 256 fp32 elements (16x16 matrix)
    float* d_out;
    hipMalloc(&d_out, N_OUT * sizeof(float));

    mfma_smoke_kernel<<<dim3(1), dim3(64)>>>(d_out);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("[mfma-smoke] kernel error: %s\n", hipGetErrorString(err));
        hipFree(d_out);
        return 1;
    }

    float h_out[N_OUT];
    hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < N_OUT; ++i) {
        if (h_out[i] != 16.0f) {
            if (errors < 4) {
                printf("[mfma-smoke] mismatch at lane=%d slot=%d: got %.3f expected 16.0\n",
                       i / 4, i % 4, h_out[i]);
            }
            ++errors;
        }
    }
    printf("[mfma-smoke] errors=%d/%d  %s\n", errors, N_OUT, errors == 0 ? "PASS" : "FAIL");
    printf("[mfma-smoke] first 8 lanes [0..3]: %.1f %.1f %.1f %.1f  %.1f %.1f %.1f %.1f\n",
           h_out[0], h_out[1], h_out[2], h_out[3],
           h_out[4], h_out[5], h_out[6], h_out[7]);

    hipFree(d_out);
    return errors == 0 ? 0 : 1;
}
