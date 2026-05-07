// Minimal VDCores-style decoupled memory/compute kernel on AMD HIP.
// Targets MI300X (gfx942). Each block has two wavefronts:
//   wave 0 = "memory core"  (global <-> LDS staging)
//   wave 1 = "compute core" (LDS -> ALU -> LDS)
// They communicate through shared LDS slots and a block barrier.
//
// Workload: c = a + b (one chunk at a time, double-buffered LDS).

#include <hip/hip_runtime.h>
#include <cmath>
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
void vdcores_demo_kernel(const float* __restrict__ a,
                         const float* __restrict__ b,
                         float* __restrict__ c) {
  __shared__ float lds_a[CHUNK];
  __shared__ float lds_b[CHUNK];
  __shared__ float lds_c[CHUNK];

  const int tid    = threadIdx.x;
  const int lane   = tid & (WAVE - 1);
  const int wave   = tid / WAVE;            // 0 = memory, 1 = compute
  const int base   = blockIdx.x * N_PER_BLK;

  for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
    const int off = chunk * CHUNK;

    // memory wavefront stages a/b global -> LDS
    if (wave == 0) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_a[i] = a[base + off + i];
        lds_b[i] = b[base + off + i];
      }
    }
    __syncthreads();

    // compute wavefront does the math in LDS
    if (wave == 1) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_c[i] = lds_a[i] + lds_b[i];
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

  std::vector<float> ha(N), hb(N), hc(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i + 1);
  }

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  HIP_CHECK(hipMalloc(&da, BYTES));
  HIP_CHECK(hipMalloc(&db, BYTES));
  HIP_CHECK(hipMalloc(&dc, BYTES));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  HIP_CHECK(hipMemcpyAsync(da, ha.data(), BYTES, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemcpyAsync(db, hb.data(), BYTES, hipMemcpyHostToDevice, stream));

  dim3 grid(N_BLOCKS), block(2 * WAVE);
  hipLaunchKernelGGL(vdcores_demo_kernel, grid, block, 0, stream, da, db, dc);

  HIP_CHECK(hipMemcpyAsync(hc.data(), dc, BYTES, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    float ref = ha[i] + hb[i];
    if (std::fabs(hc[i] - ref) > 1e-5f) ++errors;
  }
  std::printf("[vdcores-hip] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");

  hipStreamDestroy(stream);
  hipFree(da); hipFree(db); hipFree(dc);
  return errors;
}
