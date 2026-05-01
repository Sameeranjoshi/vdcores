# VDCores AMD port — L1 + L2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build L1 (async global→LDS load via `__builtin_amdgcn_global_load_lds` + `s_waitcnt vmcnt(0)`) and L2 (LDS-counter `arrive`/`wait_at_least` queue) as two new standalone HIP binaries in `app/hip/`. Each rung must print `errors=0 PASS` on MI300X.

**Architecture:** Bottom-up staircase. L0 (`vdcores_hip_demo.cpp`) is the verified baseline and is **not modified**. L1 is a verbatim copy of L0 with one surgical change — plain LDS staging is replaced by the async builtin + a `s_waitcnt` gate. L2 is a verbatim copy of L1 with one surgical change — every cross-wave `__syncthreads()` is replaced by an `LdsSignal` `arrive`/`wait_at_least` pair (the init-time `__syncthreads` stays, since it's not a producer/consumer signal). Each rung is its own binary. The Makefile gives every rung a dedicated `make run-l<N>` target so any breakage can be bisected against the rung directly below.

**Tech Stack:** HIP / hipcc, ROCm 7.2 (clang-22), gfx942 (MI300X), wave64, AMD intrinsics (`__builtin_amdgcn_global_load_lds`, `__builtin_amdgcn_s_sleep`, inline-asm `s_waitcnt vmcnt(0)`, LDS `atomicAdd`, `__threadfence_block`), SLURM batch on partition `mi3001x`.

**Spec:** `docs/superpowers/specs/2026-05-01-vdcores-amd-port-l1-l2-design.md`

---

## File structure

**Create:**
- `app/hip/vdcores_hip_l1.cpp` — L1 binary source
- `app/hip/vdcores_hip_l2.cpp` — L2 binary source
- `app/hip/run_l1_l2.sbatch` — SLURM submission script (builds all 3, runs all 3)

**Modify:**
- `app/hip/Makefile` — add `all`, L1+L2 build targets, `run-l0`/`run-l1`/`run-l2` targets, update `clean`
- `app/hip/README.md` — append L1 and L2 paragraphs

**Untouched (verify):**
- `app/hip/vdcores_hip_demo.cpp` — L0 baseline must remain byte-identical to `git HEAD`

---

## Verification model

This is HIP/GPU code with no unit-test framework. Each rung's binary contains its own host-side correctness check (`errors=N`, prints `PASS`/`FAIL`, `exit(N)`). The TDD adaptation: for every code change we re-submit `run_l1_l2.sbatch`, then `grep PASS` (and `grep FAIL`) the output. A rung is "green" only when its line shows `errors=0 PASS`. Each sbatch round is ~1 GPU-minute (build + sub-second runs).

---

### Task 1: Extend Makefile and add SLURM batch script

Set up the build/run infrastructure for all three rungs. The Makefile gets `all`, `vdcores_hip_l1`, `vdcores_hip_l2`, `run-l0`, `run-l1`, `run-l2` targets. The sbatch builds all three and runs each with its own 10s timeout. After this task we can submit and confirm L0 still passes (the only target that has a source file at this point).

**Files:**
- Modify: `app/hip/Makefile`
- Create: `app/hip/run_l1_l2.sbatch`

- [ ] **Step 1: Read current Makefile to confirm starting state**

Run: `cat app/hip/Makefile`
Expected: single-target Makefile that builds `vdcores_hip_demo` from `vdcores_hip_demo.cpp`.

- [ ] **Step 2: Replace Makefile contents**

Write `app/hip/Makefile`:

```makefile
HIPCC    ?= hipcc
# Auto-detect the local GPU arch via rocminfo; fall back to gfx942 (MI300X) if
# unavailable. Override explicitly with `make ARCH=gfx90a` for MI210, etc.
ARCH     ?= $(shell rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx/ {print $$2; exit}')
ifeq ($(strip $(ARCH)),)
ARCH := gfx942
endif
CXXFLAGS ?= -O2 -std=c++17 --offload-arch=$(ARCH) -D__AMDGCN_WAVEFRONT_SIZE=64

TARGETS := vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2

all: $(TARGETS)

vdcores_hip_demo: vdcores_hip_demo.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

vdcores_hip_l1: vdcores_hip_l1.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

vdcores_hip_l2: vdcores_hip_l2.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

run-l0: vdcores_hip_demo
	./vdcores_hip_demo

run-l1: vdcores_hip_l1
	./vdcores_hip_l1

run-l2: vdcores_hip_l2
	./vdcores_hip_l2

clean:
	rm -f $(TARGETS)

.PHONY: all run-l0 run-l1 run-l2 clean
```

- [ ] **Step 3: Create the SLURM batch script**

Write `app/hip/run_l1_l2.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=vdcores-hip
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:03:00
#SBATCH --output=vdcores-hip-%j.out
#SBATCH --error=vdcores-hip-%j.err

set -eo pipefail
REPO_DIR="/work1/maryhall/sameeran/work/vdcores"
cd "${REPO_DIR}/app/hip"

echo "=== job ${SLURM_JOB_ID} on $(hostname) ==="
date
module purge
module load rocm/7.2.0
echo "--- rocm-smi ---"
rocm-smi || true

echo "--- make clean & build (only sources that exist will compile) ---"
make clean
# Build whatever sources are present; skip missing ones gracefully.
for tgt in vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2; do
  if [ -f "${tgt}.cpp" ]; then
    echo "--- build ${tgt} ---"
    make "${tgt}"
  else
    echo "--- skip ${tgt} (no source) ---"
  fi
done

for bin in vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2; do
  if [ -x "./${bin}" ]; then
    echo "--- run ${bin} ---"
    timeout 10 "./${bin}" || echo "(${bin} exited non-zero)"
  fi
done

echo "=== job ${SLURM_JOB_ID} done ==="
date
```

- [ ] **Step 4: Submit and wait**

Run:
```bash
sbatch app/hip/run_l1_l2.sbatch
```
Capture the job ID. Wait for completion:
```bash
sacct -u $USER --format=JobID,State,Elapsed --starttime=$(date +%Y-%m-%d) | tail -5
```
Expected: job state `COMPLETED`. Look at the output:
```bash
ls -t vdcores-hip-*.out | head -1 | xargs cat
```
Expected: log shows `--- build vdcores_hip_demo ---`, `--- skip vdcores_hip_l1 (no source) ---`, `--- skip vdcores_hip_l2 (no source) ---`, then `--- run vdcores_hip_demo ---`, then `[vdcores-hip] N=32768 blocks=16 threads/block=128 errors=0 PASS`.

- [ ] **Step 5: Commit**

```bash
git add app/hip/Makefile app/hip/run_l1_l2.sbatch
git commit -m "$(cat <<'EOF'
hip-port: add L1/L2 build targets and SLURM batch script

Makefile now exposes vdcores_hip_l1 and vdcores_hip_l2 build targets plus
run-l<N> phony targets. run_l1_l2.sbatch builds whichever sources exist and
runs each binary under a 10s timeout, so the same script works at every
rung of the staircase. Existing vdcores_hip_demo (L0) build path unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create L1 source as verbatim copy of L0 with renamed print tag

Get `vdcores_hip_l1.cpp` in place. No semantic changes from L0 — only the print bracket-tag changes from `[vdcores-hip]` to `[vdcores-hip-l1]` so we can grep them apart in the same log. After this task, `make run-l1` should produce a `PASS` line that's identical to L0 except for the tag, proving the file/build wiring works before we touch any device code.

**Files:**
- Create: `app/hip/vdcores_hip_l1.cpp`

- [ ] **Step 1: Copy L0 source to L1**

Run:
```bash
cp app/hip/vdcores_hip_demo.cpp app/hip/vdcores_hip_l1.cpp
```

- [ ] **Step 2: Rename print bracket tag**

In `app/hip/vdcores_hip_l1.cpp`, replace:

```cpp
  std::printf("[vdcores-hip] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");
```

With:

```cpp
  std::printf("[vdcores-hip-l1] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");
```

(Only the bracket tag `vdcores-hip` → `vdcores-hip-l1` changes. Everything else is byte-identical to L0.)

- [ ] **Step 3: Submit batch and wait**

Run:
```bash
sbatch app/hip/run_l1_l2.sbatch
```
Wait for completion (sacct shows `COMPLETED`), then read the latest log:
```bash
ls -t vdcores-hip-*.out | head -1 | xargs cat
```

- [ ] **Step 4: Verify both L0 and L1 PASS, with the L0 baseline still byte-identical to HEAD**

In the log, expect both:
- `[vdcores-hip] N=32768 blocks=16 threads/block=128 errors=0 PASS`
- `[vdcores-hip-l1] N=32768 blocks=16 threads/block=128 errors=0 PASS`

And no `--- skip ...` lines for `vdcores_hip_l1`. Also confirm L0 source untouched:
```bash
git diff app/hip/vdcores_hip_demo.cpp
```
Expected: empty diff.

- [ ] **Step 5: Commit**

```bash
git add app/hip/vdcores_hip_l1.cpp
git commit -m "$(cat <<'EOF'
hip-port: add L1 source (verbatim copy of L0, renamed print tag)

vdcores_hip_l1.cpp is byte-identical to vdcores_hip_demo.cpp except for the
print bracket tag. Confirms the build/sbatch wiring works before any device
code changes; both L0 and L1 print errors=0 PASS in the same log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Replace plain LDS staging with `__builtin_amdgcn_global_load_lds` in L1

The actual L1 change. Inside the kernel's `if (wave == 0)` global→LDS section, swap the plain `lds_a[i] = a[...]` assignment for the async builtin, and gate completion with `s_waitcnt vmcnt(0)` before the cross-wave `__syncthreads()`. Workload also changes from `c = a + b` to `c = a` (drop `b`, drop `lds_b`, drop the `+ lds_b[i]` in compute, drop the `b` kernel arg, drop host-side `b` allocation).

**Files:**
- Modify: `app/hip/vdcores_hip_l1.cpp`

- [ ] **Step 1: Drop the `b` arg from the kernel signature and the `lds_b` shared array**

In `app/hip/vdcores_hip_l1.cpp` find:

```cpp
__global__ __launch_bounds__(2 * WAVE)
void vdcores_demo_kernel(const float* __restrict__ a,
                         const float* __restrict__ b,
                         float* __restrict__ c) {
  __shared__ float lds_a[CHUNK];
  __shared__ float lds_b[CHUNK];
  __shared__ float lds_c[CHUNK];
```

Replace with:

```cpp
__global__ __launch_bounds__(2 * WAVE)
void vdcores_l1_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
  __shared__ float lds_a[CHUNK];
  __shared__ float lds_c[CHUNK];
```

(Kernel renamed `_demo_` → `_l1_`. `b` arg removed. `lds_b` removed.)

- [ ] **Step 2: Replace the wave-0 staging loop with the async builtin**

Find:

```cpp
    // memory wavefront stages a/b global -> LDS
    if (wave == 0) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_a[i] = a[base + off + i];
        lds_b[i] = b[base + off + i];
      }
    }
    __syncthreads();
```

Replace with:

```cpp
    // memory wavefront stages a global -> LDS via async load
    if (wave == 0) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        __builtin_amdgcn_global_load_lds(
            /*global src*/ &a[base + off + i],
            /*lds   dst*/ &lds_a[i],
            /*size  bytes*/ sizeof(float),
            /*offset*/ 0,
            /*aux*/ 0);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    __syncthreads();
```

- [ ] **Step 3: Drop `+ lds_b[i]` in the compute wave (workload becomes `c = a`)**

Find:

```cpp
    // compute wavefront does the math in LDS
    if (wave == 1) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_c[i] = lds_a[i] + lds_b[i];
      }
    }
    __syncthreads();
```

Replace with:

```cpp
    // compute wavefront passes data through LDS (workload is pure copy)
    if (wave == 1) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_c[i] = lds_a[i];
      }
    }
    __syncthreads();
```

- [ ] **Step 4: Update the host-side launch and reference**

Find:

```cpp
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
  std::printf("[vdcores-hip-l1] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");

  hipStreamDestroy(stream);
  hipFree(da); hipFree(db); hipFree(dc);
  return errors;
}
```

Replace with:

```cpp
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
```

- [ ] **Step 5: Submit batch, wait, read log**

Run:
```bash
sbatch app/hip/run_l1_l2.sbatch
```
Wait for `COMPLETED`, then:
```bash
ls -t vdcores-hip-*.out | head -1 | xargs cat
```

- [ ] **Step 6: Verify L1 PASS**

Expect in the log:
- `[vdcores-hip] N=32768 ... errors=0 PASS` (L0 baseline still green)
- `[vdcores-hip-l1] N=32768 ... errors=0 PASS` (L1 now uses async builtin)

If the build instead fails with `error: use of undeclared identifier '__builtin_amdgcn_global_load_lds'` or similar, the builtin name has drifted in this clang version. Fallback: replace the builtin call in step 2 with this inline-asm form (identical semantics, different surface):

```cpp
      for (int i = lane; i < CHUNK; i += WAVE) {
        unsigned lds_off = static_cast<unsigned>(
            __builtin_amdgcn_ds_bpermute(0, 0) + 0);  // touch ds to force m0 setup
        (void)lds_off;
        // Set m0 = lds_offset (in bytes) for global_load_lds_dword target.
        unsigned m0_val = static_cast<unsigned>(
            reinterpret_cast<uintptr_t>(&lds_a[i]) & 0xFFFFu);
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "global_load_lds_dword %1, off\n\t"
            :
            : "s"(m0_val), "v"(&a[base + off + i])
            : "memory", "m0");
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
```

If this fallback is taken, note it in the commit message so we know the builtin path needs revisiting. Most likely the builtin works as written and the fallback is unused.

If the build passes but L1 prints `FAIL` with non-zero errors, do **not** patch around it — that's a real correctness bug in the L1 surface. Stop the plan, investigate, and only proceed once L1 is `PASS`.

- [ ] **Step 7: Commit**

```bash
git add app/hip/vdcores_hip_l1.cpp
git commit -m "$(cat <<'EOF'
hip-port: L1 — async global→LDS load via __builtin_amdgcn_global_load_lds

Wave 0's plain LDS staging loop (lds_a[i] = a[..]) is replaced with the
gfx942-native __builtin_amdgcn_global_load_lds intrinsic + s_waitcnt vmcnt(0)
gate. Workload simplified from c = a + b to c = a (lds_b dropped; compute
wave is now a trivial passthrough). Cross-wave synchronisation still uses
__syncthreads — only the load primitive changed.

Verified: errors=0 PASS on MI300X.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Create L2 source as verbatim copy of L1 with renamed tag

Same pattern as Task 2 but starting from L1 instead of L0. Get `vdcores_hip_l2.cpp` in place with only the print bracket tag and kernel name changed. Verify all three rungs still PASS in one batch.

**Files:**
- Create: `app/hip/vdcores_hip_l2.cpp`

- [ ] **Step 1: Copy L1 source to L2**

Run:
```bash
cp app/hip/vdcores_hip_l1.cpp app/hip/vdcores_hip_l2.cpp
```

- [ ] **Step 2: Rename kernel `_l1_` → `_l2_`**

In `app/hip/vdcores_hip_l2.cpp`, find:

```cpp
__global__ __launch_bounds__(2 * WAVE)
void vdcores_l1_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
```

Replace with:

```cpp
__global__ __launch_bounds__(2 * WAVE)
void vdcores_l2_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
```

And in the launch site, find:

```cpp
  hipLaunchKernelGGL(vdcores_l1_kernel, grid, block, 0, stream, da, dc);
```

Replace with:

```cpp
  hipLaunchKernelGGL(vdcores_l2_kernel, grid, block, 0, stream, da, dc);
```

- [ ] **Step 3: Rename print bracket tag `vdcores-hip-l1` → `vdcores-hip-l2`**

Find:

```cpp
  std::printf("[vdcores-hip-l1] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");
```

Replace with:

```cpp
  std::printf("[vdcores-hip-l2] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");
```

- [ ] **Step 4: Submit batch, wait, read log**

Run:
```bash
sbatch app/hip/run_l1_l2.sbatch
```
Wait for `COMPLETED`. Read the latest log:
```bash
ls -t vdcores-hip-*.out | head -1 | xargs cat
```

- [ ] **Step 5: Verify all three PASS**

Expect:
- `[vdcores-hip] N=32768 ... errors=0 PASS`
- `[vdcores-hip-l1] N=32768 ... errors=0 PASS`
- `[vdcores-hip-l2] N=32768 ... errors=0 PASS`

L0 source must still be byte-identical to HEAD:
```bash
git diff app/hip/vdcores_hip_demo.cpp
```
Expected: empty.

- [ ] **Step 6: Commit**

```bash
git add app/hip/vdcores_hip_l2.cpp
git commit -m "$(cat <<'EOF'
hip-port: add L2 source (verbatim copy of L1, renamed kernel + print tag)

vdcores_hip_l2.cpp is byte-identical to vdcores_hip_l1.cpp except for the
kernel name (vdcores_l2_kernel) and print tag ([vdcores-hip-l2]). Confirms
the wiring before the LDS-counter queue swap; all three rungs print
errors=0 PASS in the same log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Replace `__syncthreads()` between waves with `LdsSignal` `arrive` / `wait_at_least` in L2

The actual L2 change. Add the `LdsSignal` struct and the `arrive` / `wait_at_least` helpers at file scope. Add two `__shared__` signals (`loaded`, `computed`) inside the kernel, init them once, and replace each cross-wave `__syncthreads()` with the appropriate `arrive` (producer) / `wait_at_least` (consumer) pair.

**Files:**
- Modify: `app/hip/vdcores_hip_l2.cpp`

- [ ] **Step 1: Add the `LdsSignal` primitives at file scope**

In `app/hip/vdcores_hip_l2.cpp`, find the line:

```cpp
constexpr int WAVE      = 64;     // AMD wavefront
```

Insert directly **before** that line (i.e. between the `#define HIP_CHECK(...)` block and the `constexpr int WAVE` line):

```cpp
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
    atomicAdd(&s.counter, 1u);
    __threadfence_block();
  }
}

__device__ __forceinline__
void wait_at_least(LdsSignal& s, unsigned target) {
  while (atomicAdd(&s.counter, 0u) < target) {   // atomic-read forces fresh fetch
    __builtin_amdgcn_s_sleep(1);
  }
}

```

(Trailing blank line before the existing `constexpr int WAVE` line.)

- [ ] **Step 2: Add `__shared__` signals and an init barrier**

In the kernel, find:

```cpp
__global__ __launch_bounds__(2 * WAVE)
void vdcores_l2_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
  __shared__ float lds_a[CHUNK];
  __shared__ float lds_c[CHUNK];

  const int tid    = threadIdx.x;
  const int lane   = tid & (WAVE - 1);
  const int wave   = tid / WAVE;            // 0 = memory, 1 = compute
  const int base   = blockIdx.x * N_PER_BLK;
```

Replace with:

```cpp
__global__ __launch_bounds__(2 * WAVE)
void vdcores_l2_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
  __shared__ float lds_a[CHUNK];
  __shared__ float lds_c[CHUNK];
  __shared__ LdsSignal loaded;     // mem-wave  → compute-wave
  __shared__ LdsSignal computed;   // compute-wave → mem-wave (store)

  const int tid    = threadIdx.x;
  const int lane   = tid & (WAVE - 1);
  const int wave   = tid / WAVE;            // 0 = memory, 1 = compute
  const int base   = blockIdx.x * N_PER_BLK;

  // Initialise both counters to 0 once. This __syncthreads is an init
  // barrier, not a producer/consumer signal — we keep it.
  if (tid == 0) {
    loaded.counter   = 0;
    computed.counter = 0;
  }
  __syncthreads();
```

- [ ] **Step 3: Replace the chunk loop body — swap the three cross-wave `__syncthreads` for arrive/wait pairs**

Find the entire chunk loop:

```cpp
  for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
    const int off = chunk * CHUNK;

    // memory wavefront stages a global -> LDS via async load
    if (wave == 0) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        __builtin_amdgcn_global_load_lds(
            /*global src*/ &a[base + off + i],
            /*lds   dst*/ &lds_a[i],
            /*size  bytes*/ sizeof(float),
            /*offset*/ 0,
            /*aux*/ 0);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
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
```

Replace with:

```cpp
  for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
    const int off = chunk * CHUNK;
    const unsigned target = static_cast<unsigned>(chunk + 1);

    // memory wavefront stages a global -> LDS via async load, then signals.
    if (wave == 0) {
      for (int i = lane; i < CHUNK; i += WAVE) {
        __builtin_amdgcn_global_load_lds(
            /*global src*/ &a[base + off + i],
            /*lds   dst*/ &lds_a[i],
            /*size  bytes*/ sizeof(float),
            /*offset*/ 0,
            /*aux*/ 0);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      arrive(loaded);                                    // signal 1: load done
    }

    // compute wavefront waits for data, copies LDS->LDS, then signals.
    if (wave == 1) {
      wait_at_least(loaded, target);                     // wait for signal 1
      for (int i = lane; i < CHUNK; i += WAVE) {
        lds_c[i] = lds_a[i];
      }
      arrive(computed);                                  // signal 2: compute done
    }

    // memory wavefront waits for compute, drains LDS -> global.
    if (wave == 0) {
      wait_at_least(computed, target);                   // wait for signal 2
      for (int i = lane; i < CHUNK; i += WAVE) {
        c[base + off + i] = lds_c[i];
      }
    }
  }
```

(Three `__syncthreads()` calls between waves are gone; the trailing one at the end of the loop body is also gone — wave 0's `wait_at_least(computed, target)` already orders against wave 1's `arrive(computed)`, and the next chunk's wave-1 entry will wait again on `loaded`. The init `__syncthreads` at the top of the kernel is the only one left.)

- [ ] **Step 4: Submit batch, wait, read log**

Run:
```bash
sbatch app/hip/run_l1_l2.sbatch
```
Wait for `COMPLETED`. Read the latest log:
```bash
ls -t vdcores-hip-*.out | head -1 | xargs cat
```

- [ ] **Step 5: Verify all three rungs PASS**

Expect, in order:
- `[vdcores-hip] N=32768 blocks=16 threads/block=128 errors=0 PASS`
- `[vdcores-hip-l1] N=32768 blocks=16 threads/block=128 errors=0 PASS`
- `[vdcores-hip-l2] N=32768 blocks=16 threads/block=128 errors=0 PASS`

If L0 and L1 still PASS but L2 fails (compile error, hang, or `errors=N FAIL`): the bug is in this task's diff. The diff between `vdcores_hip_l1.cpp` and `vdcores_hip_l2.cpp` is exactly the `LdsSignal` block, the two `__shared__` signal vars, the init barrier, and the three signal calls. That's the suspect set — bisect there.

Specifically, if L2 hangs (timeout 10 fires, `(vdcores_hip_l2 exited non-zero)` printed), the most likely cause is the spin-poll never converging — check that `arrive` and `wait_at_least` see the same `LdsSignal` instance (they should, both refer to `__shared__ loaded` / `__shared__ computed`), and that `target` is `chunk + 1` (not `chunk`).

If the build fails citing `atomicAdd` on `__shared__`: cast the LDS pointer explicitly: `atomicAdd(reinterpret_cast<unsigned*>(&s.counter), 1u);`. AMD's HIP atomicAdd should accept generic pointers, but some clang versions need the cast.

- [ ] **Step 6: Commit**

```bash
git add app/hip/vdcores_hip_l2.cpp
git commit -m "$(cat <<'EOF'
hip-port: L2 — replace __syncthreads with LDS-counter signals

Adds LdsSignal struct + arrive(s) / wait_at_least(s, target) primitives. The
kernel now carries two __shared__ signals (loaded, computed) initialised once
under an init __syncthreads. All three cross-wave __syncthreads in the chunk
loop are replaced with explicit arrive (on the producer side) and
wait_at_least (on the consumer side) pairs. atomicAdd(p, 0) is used for the
spin-poll read so the LLVM backend cannot hoist the load past s_sleep.

Verified: errors=0 PASS on MI300X.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Update `app/hip/README.md` with L1 and L2 paragraphs

Document the staircase so a fresh reader knows what each binary tests and what `PASS` output to expect.

**Files:**
- Modify: `app/hip/README.md`

- [ ] **Step 1: Read current README**

Run: `cat app/hip/README.md`
Expected: existing README describes L0 only.

- [ ] **Step 2: Append L1+L2 sections**

Append to `app/hip/README.md` (after the existing "What's intentionally NOT here" section):

```markdown

## Staircase rungs

The directory now contains a staircase of three binaries, each one surgical change from the rung below. The staircase is the debugger: if a rung regresses, the diff against the rung below it is the suspect list.

### L0 — `vdcores_hip_demo`

Already documented above. Plain LDS staging + `__syncthreads`. Workload `c = a + b`.

### L1 — `vdcores_hip_l1`

Same skeleton as L0, but wave 0's global→LDS staging uses the AMD-native async builtin:

```cpp
__builtin_amdgcn_global_load_lds(global_ptr, lds_ptr, sizeof(float), 0, 0);
asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
```

Workload simplified to `c = a` (pure copy — no compute work). Cross-wave sync still uses `__syncthreads`. Build + run:

```bash
make run-l1
```

Expected: `[vdcores-hip-l1] N=32768 blocks=16 threads/block=128 errors=0 PASS`

### L2 — `vdcores_hip_l2`

Same async load as L1. The three `__syncthreads` between waves are replaced with an LDS-counter `arrive(s) / wait_at_least(s, target)` queue. Two signals per CTA: `loaded` (mem→compute) and `computed` (compute→mem-store). Single buffer, monotonic counters. Build + run:

```bash
make run-l2
```

Expected: `[vdcores-hip-l2] N=32768 blocks=16 threads/block=128 errors=0 PASS`

### Running all three on MI300X via SLURM

```bash
sbatch run_l1_l2.sbatch
```

The job builds whichever sources are present and runs each binary under a 10s timeout. All three lines should print `errors=0 PASS`.
```

- [ ] **Step 3: Commit**

```bash
git add app/hip/README.md
git commit -m "$(cat <<'EOF'
hip-port: document the L0/L1/L2 staircase in app/hip/README

Adds a Staircase rungs section explaining what L1 (async global→LDS load) and
L2 (LDS-counter queue) replace, the expected PASS line for each, and how to
submit the SLURM batch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Final verification ladder + close-out

Run the canonical verification command, confirm all three PASS in one batch, sanity-check that L0 was never modified, then mark the spec complete.

**Files:**
- (no source changes; just verification)

- [ ] **Step 1: Confirm L0 source untouched since the start of v1**

Run:
```bash
git log --oneline -- app/hip/vdcores_hip_demo.cpp
```
Expected: most recent commit touching this file is older than today's spec commit (i.e. the file hasn't been modified by Tasks 1–6).

```bash
git diff main...HEAD -- app/hip/vdcores_hip_demo.cpp
```
Expected: empty diff.

- [ ] **Step 2: Submit final verification sbatch**

Run:
```bash
sbatch app/hip/run_l1_l2.sbatch
```
Wait for `COMPLETED`.

- [ ] **Step 3: Read log and confirm all three PASS**

Run:
```bash
ls -t vdcores-hip-*.out | head -1 | xargs cat | grep -E "vdcores-hip(-l[12])?\] N="
```
Expected (three lines):
```
[vdcores-hip] N=32768 blocks=16 threads/block=128  errors=0  PASS
[vdcores-hip-l1] N=32768 blocks=16 threads/block=128  errors=0  PASS
[vdcores-hip-l2] N=32768 blocks=16 threads/block=128  errors=0  PASS
```

If any line is missing or shows `FAIL` / non-zero `errors=N`, that rung is not green. Stop the plan, investigate that rung's diff against the rung below, and only declare v1 done after all three are `PASS` in a single sbatch run.

- [ ] **Step 4: Tag the spec done**

Run:
```bash
git tag -a vdcores-amd-port-l1-l2-v1 -m "VDCores AMD port: L1+L2 verified PASS on MI300X"
```

(The tag is local; not pushed unless the user asks.)

- [ ] **Step 5: Final commit (if anything stray remains)**

Run `git status` to confirm a clean tree. If any uncommitted changes remain (e.g. a `.gitignore` tweak or stray sbatch output you want versioned), stage and commit them now with a `chore:` message. Otherwise this step is a no-op.

---

## Self-review checklist

- **Spec coverage:** every requirement in the spec maps to a task — Makefile/build infra (Task 1), L1 source creation (Task 2), L1 async-load swap (Task 3), L2 source creation (Task 4), L2 queue swap (Task 5), README update (Task 6), final verification ladder (Task 7). The "out of scope" list in the spec is respected: no throughput timing, no double-buffering, no MFMA, no Python, no megakernel touched.
- **Placeholders:** none. Every step has actual code or commands.
- **Type/name consistency:** kernel names `vdcores_l1_kernel` (Task 3) and `vdcores_l2_kernel` (Task 4 step 2) are referenced consistently in their respective `hipLaunchKernelGGL` sites. `LdsSignal::counter` is used consistently in `arrive` and `wait_at_least`. `target = chunk + 1` matches the `arrive` increment of `1` per chunk.
- **Verification:** every code-changing task ends with an sbatch round and a grep against the expected PASS line. The staircase property holds — if a later rung regresses, only its diff against the rung below is in scope.
