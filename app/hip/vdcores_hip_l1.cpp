// Minimal VDCores-style decoupled memory/compute kernel on AMD HIP.
// Targets MI300X (gfx942). Each block has two wavefronts:
//   wave 0 = "memory core"  (global <-> LDS staging)
//   wave 1 = "compute core" (LDS -> ALU -> LDS)
// They communicate through shared LDS slots and a block barrier.
//
// Workload: c = a (one chunk at a time, single-buffered LDS).

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                       \
  do {                                                                     \
    hipError_t _e = (x);                                                   \
    if (_e != hipSuccess) {                                                \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                      \
                   hipGetErrorString(_e), __FILE__, __LINE__);             \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

constexpr int WAVE      = 64;     // AMD wavefront
constexpr int CHUNK     = 256;    // floats per LDS slot
constexpr int N_CHUNKS  = 8;      // chunks each block processes
constexpr int N_PER_BLK = CHUNK * N_CHUNKS;

__global__ __launch_bounds__(2 * WAVE)
void vdcores_l1_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
  __shared__ float lds_a[CHUNK];
  __shared__ float lds_c[CHUNK];

  const int tid    = threadIdx.x;
  const int lane   = tid & (WAVE - 1);
  const int wave   = tid / WAVE;            // 0 = memory, 1 = compute
  const int base   = blockIdx.x * N_PER_BLK;

  for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
    const int off = chunk * CHUNK;

    // memory wavefront stages a global -> LDS via explicit async load sequence.
    // Phase 1: post all VMEM loads into VGPRs (flight simultaneously).
    // Phase 2: s_waitcnt vmcnt(0) — drain the VMEM pipeline.
    // Phase 3: ds_write_b32 each VGPR into its LDS slot.
    // (global_load_lds_dword builtin crashes ROCm 7.2 backend for gfx942;
    //  spec inline-asm fallback also fails — SGPR/VGPR constraint on m0.)
    if (wave == 0) {
      static_assert(CHUNK / WAVE == 4,
          "flat_load/ds_write unroll assumes CHUNK/WAVE==4; update Phase 1-3 if constants change");
      // CHUNK/WAVE = 256/64 = 4 iterations per lane
      float tmp0, tmp1, tmp2, tmp3;
      typedef __attribute__((address_space(3))) float* lds_ptr_t;
      // Phase 1: issue all four VMEM loads
      asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp0) : "v"(&a[base + off + lane +      0]) : "memory");
      asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp1) : "v"(&a[base + off + lane + WAVE]) : "memory");
      asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp2) : "v"(&a[base + off + lane + 2*WAVE]) : "memory");
      asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp3) : "v"(&a[base + off + lane + 3*WAVE]) : "memory");
      // Phase 2: wait for all VMEM loads to land in VGPRs
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      // Phase 3: write VGPRs into LDS
      uint32_t d0 = (uint32_t)(uintptr_t)(lds_ptr_t)(&lds_a[lane +      0]);
      uint32_t d1 = (uint32_t)(uintptr_t)(lds_ptr_t)(&lds_a[lane + WAVE]);
      uint32_t d2 = (uint32_t)(uintptr_t)(lds_ptr_t)(&lds_a[lane + 2*WAVE]);
      uint32_t d3 = (uint32_t)(uintptr_t)(lds_ptr_t)(&lds_a[lane + 3*WAVE]);
      asm volatile("ds_write_b32 %0, %1" :: "v"(d0), "v"(tmp0) : "memory");
      asm volatile("ds_write_b32 %0, %1" :: "v"(d1), "v"(tmp1) : "memory");
      asm volatile("ds_write_b32 %0, %1" :: "v"(d2), "v"(tmp2) : "memory");
      asm volatile("ds_write_b32 %0, %1" :: "v"(d3), "v"(tmp3) : "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    }
    __syncthreads();

    // compute wavefront passes data through LDS (workload is pure copy)
    if (wave == 1) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_c[i] = lds_a[i];
      }
    }
    __syncthreads();

    // memory wavefront drains LDS -> global
    if (wave == 0) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        c[base + off + i] = lds_c[i];
      }
    }
    __syncthreads();
  }
}

int main() {
  constexpr int N_BLOCKS = 16;
  constexpr int N        = N_PER_BLK * N_BLOCKS;
  constexpr size_t BYTES = N * sizeof(float);

  std::vector<float> ha(N), hc(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    ha[i] = static_cast<float>(i);
  }

  float *da = nullptr, *dc = nullptr;
  HIP_CHECK(hipMalloc(&da, BYTES));
  HIP_CHECK(hipMalloc(&dc, BYTES));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  HIP_CHECK(hipMemcpyAsync(da, ha.data(), BYTES, hipMemcpyHostToDevice, stream));

  dim3 grid(N_BLOCKS), block(2 * WAVE);
  hipLaunchKernelGGL(vdcores_l1_kernel, grid, block, 0, stream, da, dc);

  HIP_CHECK(hipMemcpyAsync(hc.data(), dc, BYTES, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    float ref = ha[i];                       // workload is c = a
    if (std::fabs(hc[i] - ref) > 1e-5f) ++errors;
  }
  std::printf("[vdcores-hip-l1] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");

  hipStreamDestroy(stream);
  hipFree(da); hipFree(dc);
  return errors;
}
