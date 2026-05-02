// AMD-native analogue of dae_copy_smoke.py.
//
// Workload mirrors the Python script verbatim:
//   1 block × 1 chunk × 256 floats (= 1024 bytes per chunk = num_sms*num_loads*load_bytes)
//   vec = arange(256, dtype=float32)
//   out = zeros_like(vec)
//   copy vec → out via the DAE 3-wave pipeline (LD wave + compute wave + ST wave)
//   assert torch.equal(vec, out) → prints "equal: True" / "equal: False"
//
// Engine is the L3 staircase rung (3-wave, single buffer, LDS-counter signals,
// explicit s_waitcnt discipline on AMD wave64). Same kernel topology, downsized
// to dae_copy_smoke.py's 256-element workload.
//
// Print format mirrors the Python script for grep parity:
//   equal: True
//   vec[:8]: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
//   out[:8]: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

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

// ---------- LDS-counter signal primitive (lifted from L2/L3) ----------
struct LdsSignal {
  unsigned counter;
};

__device__ __forceinline__
void arrive(LdsSignal& s) {
  if ((threadIdx.x & 63) == 0) {
    __threadfence_block();
    atomicAdd(&s.counter, 1u);
  }
}

__device__ __forceinline__
void wait_at_least(LdsSignal& s, unsigned target) {
  while (atomicAdd(&s.counter, 0u) < target) {
    __builtin_amdgcn_s_sleep(1);
  }
}

constexpr int WAVE      = 64;     // AMD wave64 (gfx942 has no wave32)
constexpr int CHUNK     = 256;    // floats per LDS slot — matches load_bytes/4
constexpr int N_CHUNKS  = 1;      // num_loads from the Python smoke
constexpr int N_PER_BLK = CHUNK * N_CHUNKS;

// 3-wave LD/compute/ST kernel. Same structure as vdcores_hip_l3.cpp:
//   wave 0 = LD       (global a → lds_a)
//   wave 1 = compute  (lds_a → lds_c, trivial passthrough since c = a)
//   wave 2 = ST       (lds_c → global c)
// Three signals (loaded, computed, stored) close the producer/consumer ring.
__global__ __launch_bounds__(3 * WAVE)
void dae_copy_smoke_kernel(const float* __restrict__ a,
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

  if (tid == 0) {
    loaded.counter   = 0;
    computed.counter = 0;
    stored.counter   = 0;
  }
  __syncthreads();

  for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
    const int off = chunk * CHUNK;
    const unsigned target = static_cast<unsigned>(chunk + 1);

    // wave 0 (LD): wait for slot to be free, then async-load.
    if (wave == 0) {
      wait_at_least(stored, static_cast<unsigned>(chunk));
      static_assert(CHUNK / WAVE == 4,
          "flat_load/ds_write unroll assumes CHUNK/WAVE==4; update Phase 1-3 if constants change");
      float tmp0, tmp1, tmp2, tmp3;
      typedef __attribute__((address_space(3))) float* lds_ptr_t;
      // Phase 1: post all four VMEM loads simultaneously.
      asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp0) : "v"(&a[base + off + lane +      0]) : "memory");
      asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp1) : "v"(&a[base + off + lane + WAVE]) : "memory");
      asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp2) : "v"(&a[base + off + lane + 2*WAVE]) : "memory");
      asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp3) : "v"(&a[base + off + lane + 3*WAVE]) : "memory");
      // Phase 2: drain VMEM pipeline.
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      // Phase 3: write VGPRs into LDS.
      uint32_t d0 = (uint32_t)(uintptr_t)(lds_ptr_t)(&lds_a[lane +      0]);
      uint32_t d1 = (uint32_t)(uintptr_t)(lds_ptr_t)(&lds_a[lane + WAVE]);
      uint32_t d2 = (uint32_t)(uintptr_t)(lds_ptr_t)(&lds_a[lane + 2*WAVE]);
      uint32_t d3 = (uint32_t)(uintptr_t)(lds_ptr_t)(&lds_a[lane + 3*WAVE]);
      asm volatile("ds_write_b32 %0, %1" :: "v"(d0), "v"(tmp0) : "memory");
      asm volatile("ds_write_b32 %0, %1" :: "v"(d1), "v"(tmp1) : "memory");
      asm volatile("ds_write_b32 %0, %1" :: "v"(d2), "v"(tmp2) : "memory");
      asm volatile("ds_write_b32 %0, %1" :: "v"(d3), "v"(tmp3) : "memory");
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      arrive(loaded);
    }

    // wave 1 (compute): wait for data, copy LDS->LDS (workload is pure copy).
    if (wave == 1) {
      wait_at_least(loaded, target);
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_c[i] = lds_a[i];
      }
      arrive(computed);
    }

    // wave 2 (ST): wait for compute, drain LDS -> global, signal slot free.
    if (wave == 2) {
      wait_at_least(computed, target);
      for (int i = lane; i < CHUNK; i += WAVE) {
        c[base + off + i] = lds_c[i];
      }
      arrive(stored);
    }
  }
}

// Host-side helper: print "[v0, v1, ..., v7]" exactly like Python's list repr
// (one decimal place, ", " separator, square brackets) so the smoke output is
// grep-comparable to dae_copy_smoke.py's.
static void print_first_8_like_python(const char* name, const float* v) {
  std::printf("%s [", name);
  for (int i = 0; i < 8; ++i) {
    std::printf("%.1f%s", v[i], i == 7 ? "" : ", ");
  }
  std::printf("]\n");
}

int main() {
  // Match dae_copy_smoke.py: num_sms=1, num_loads=1, load_bytes=1024.
  constexpr int  num_sms     = 1;
  constexpr int  num_loads   = 1;
  constexpr int  load_bytes  = 1024;
  constexpr int  N           = num_sms * num_loads * (load_bytes / 4);   // 256
  constexpr size_t BYTES     = static_cast<size_t>(N) * sizeof(float);

  static_assert(N == num_sms * N_PER_BLK,
                "Python workload shape must match the kernel's N_BLOCKS*N_PER_BLK");

  // vec = torch.arange(256, dtype=float32) on the host.
  std::vector<float> vec(N), out(N, 0.0f);
  for (int i = 0; i < N; ++i) {
    vec[i] = static_cast<float>(i);
  }

  float *d_vec = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_vec, BYTES));
  HIP_CHECK(hipMalloc(&d_out, BYTES));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  HIP_CHECK(hipMemcpyAsync(d_vec, vec.data(), BYTES, hipMemcpyHostToDevice, stream));
  // out is initialized to zeros on the device side (hipMalloc + memset).
  HIP_CHECK(hipMemsetAsync(d_out, 0, BYTES, stream));

  // Launch: 1 block (= 1 SM in the Python sense), 3 * WAVE = 192 threads.
  dim3 grid(num_sms), block(3 * WAVE);
  hipLaunchKernelGGL(dae_copy_smoke_kernel, grid, block, 0, stream, d_vec, d_out);

  HIP_CHECK(hipMemcpyAsync(out.data(), d_out, BYTES, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // torch.equal-equivalent: bitwise compare. For the arange→copy workload, exact
  // equality is correct (no float arithmetic happens in the kernel — pure mov).
  bool equal = true;
  for (int i = 0; i < N; ++i) {
    if (vec[i] != out[i]) { equal = false; break; }
  }

  std::printf("equal: %s\n", equal ? "True" : "False");
  print_first_8_like_python("vec[:8]:", vec.data());
  print_first_8_like_python("out[:8]:", out.data());

  hipStreamDestroy(stream);
  hipFree(d_vec); hipFree(d_out);
  return equal ? 0 : 1;
}
