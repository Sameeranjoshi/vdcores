// Minimal VDCores-style decoupled memory/compute kernel on AMD HIP.
// Targets MI300X (gfx942). Each block has two wavefronts:
//   wave 0 = "memory core"  (global <-> LDS staging)
//   wave 1 = "compute core" (LDS -> ALU -> LDS)
// They communicate through shared LDS slots and an LDS-counter signal pair.
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

// ---------- L2: LDS-counter signal primitive ----------
//
// One LDS word per signal. Producer increments via atomicAdd; consumer spins
// on atomic-read until the counter has reached its target. atomicAdd on LDS
// is the bulletproof spin-poll on AMD HIP — a plain or volatile load can be
// hoisted past s_sleep by the backend, but an atomic op is a hard memory
// barrier that forces a fresh fetch from LDS each iteration.
//
struct LdsSignal {
  unsigned counter;
};

__device__ __forceinline__
void arrive(LdsSignal& s) {
  if ((threadIdx.x & 63) == 0) {                 // one lane per wave signals
    __threadfence_block();                        // flush prior LDS writes
    atomicAdd(&s.counter, 1u);                    // publish the signal
  }
}

__device__ __forceinline__
void wait_at_least(LdsSignal& s, unsigned target) {
  while (atomicAdd(&s.counter, 0u) < target) {   // atomic-read forces fresh fetch
    __builtin_amdgcn_s_sleep(1);
  }
}

constexpr int WAVE      = 64;     // AMD wavefront
constexpr int CHUNK     = 256;    // floats per LDS slot
constexpr int N_CHUNKS  = 8;      // chunks each block processes
constexpr int N_PER_BLK = CHUNK * N_CHUNKS;

__global__ __launch_bounds__(3 * WAVE)
void vdcores_l3_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
  __shared__ float lds_a[CHUNK];
  __shared__ float lds_c[CHUNK];
  __shared__ LdsSignal loaded;     // LD wave  → compute wave
  __shared__ LdsSignal computed;   // compute  → ST wave
  __shared__ LdsSignal stored;     // ST       → LD wave (closes ring; gates slot reuse)

  const int tid    = threadIdx.x;
  const int lane   = tid & (WAVE - 1);
  const int wave   = tid / WAVE;            // 0 = LD, 1 = compute, 2 = ST
  const int base   = blockIdx.x * N_PER_BLK;

  // Initialise all three counters to 0 once. This __syncthreads is an
  // init barrier, not a producer/consumer signal — we keep it.
  if (tid == 0) {
    loaded.counter   = 0;
    computed.counter = 0;
    stored.counter   = 0;
  }
  __syncthreads();

  for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
    const int off = chunk * CHUNK;
    const unsigned target = static_cast<unsigned>(chunk + 1);

    // wave 0 (LD): wait for the slot to be free (prev ST done), then async-load.
    // (global_load_lds_dword builtin crashes ROCm 7.2 backend for gfx942;
    //  spec inline-asm fallback also fails — SGPR/VGPR constraint on m0.)
    // Phase 1: post all VMEM loads into VGPRs (flight simultaneously).
    // Phase 2: s_waitcnt vmcnt(0) — drain the VMEM pipeline.
    // Phase 3: ds_write_b32 each VGPR into its LDS slot.
    if (wave == 0) {
      wait_at_least(stored, static_cast<unsigned>(chunk));   // gate on slot reuse
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
      arrive(loaded);                                        // signal: load done
    }

    // wave 1 (compute): wait for data, copy LDS->LDS, signal compute done.
    if (wave == 1) {
      wait_at_least(loaded, target);
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_c[i] = lds_a[i];
      }
      arrive(computed);
    }

    // wave 2 (ST): wait for compute, drain LDS -> global, signal store done
    // so wave 0 can reuse the slot for chunk+1.
    if (wave == 2) {
      wait_at_least(computed, target);
      for (int i = lane; i < CHUNK; i += WAVE) {
        c[base + off + i] = lds_c[i];
      }
      arrive(stored);
    }
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

  dim3 grid(N_BLOCKS), block(3 * WAVE);
  hipLaunchKernelGGL(vdcores_l3_kernel, grid, block, 0, stream, da, dc);

  HIP_CHECK(hipMemcpyAsync(hc.data(), dc, BYTES, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    float ref = ha[i];                       // workload is c = a
    if (std::fabs(hc[i] - ref) > 1e-5f) ++errors;
  }
  std::printf("[vdcores-hip-l3] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 3 * WAVE, errors, errors ? "FAIL" : "PASS");

  hipStreamDestroy(stream);
  hipFree(da); hipFree(dc);
  return errors;
}
