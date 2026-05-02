# VDCores AMD port — L3 Implementation Plan

> **For agentic workers:** this plan is being executed in-session by the controller (not via dispatched subagents) per the user's "minimize GPU hours" directive. All compile-checks happen on the login node via local `hipcc`; one sbatch run is submitted at the very end to verify all four rungs PASS in one shot.

**Goal:** Add L3 to the staircase — a 3-wavefront variant of L2 where the `mem` wave is split into `LD` and `ST` waves with the compute wave between them. Single buffer; new `stored` signal closes the producer/consumer ring. Each prior rung (L0/L1/L2) stays byte-identical.

**Architecture:** Surgical change to `vdcores_hip_l2.cpp`: bump block size from `2*WAVE` to `3*WAVE`, add a 3rd `__shared__ LdsSignal` (`stored`), refactor the chunk loop so LD waits on `stored` (slot-free) before each iteration, compute waits on `loaded`, and ST (the new wave 2) waits on `computed` then arrives `stored` to close the loop.

**Tech Stack:** Same as L2. HIP/hipcc, ROCm 7.2 (clang-22), gfx942 (MI300X), wave64. No new primitives.

**Spec:** `docs/superpowers/specs/2026-05-01-vdcores-amd-port-l3-design.md`

---

## File structure

**Create:**
- `app/hip/vdcores_hip_l3.cpp` — L3 binary source (~190 lines)

**Modify:**
- `app/hip/Makefile` — add `vdcores_hip_l3` build target, `run-l3` phony, extend `TARGETS`
- `app/hip/run_l1_l2.sbatch` → **rename to** `app/hip/run_staircase.sbatch`; update build/run loops to include `vdcores_hip_l3`
- `app/hip/README.md` — append L3 section

**Untouched (verify):**
- `app/hip/vdcores_hip_demo.cpp` — L0
- `app/hip/vdcores_hip_l1.cpp` — L1
- `app/hip/vdcores_hip_l2.cpp` — L2
- All three are tagged baselines from `vdcores-amd-port-l1-l2-v1` and must remain byte-identical to that tag.

---

## Verification model

This rung's verification differs from L1+L2 in a deliberate way: **no per-task sbatch**, only **one sbatch at the very end**. Each task instead does a local `hipcc` compile-check on the login node (no GPU touched) to catch syntax/build errors immediately. The end-of-pass sbatch verifies all four rungs (L0/L1/L2/L3) in a single job.

Local compile-check command per task: `cd app/hip && make vdcores_hip_l3 ARCH=gfx942` (works on the login node since hipcc compilation is GPU-free).

---

### Task 1: Extend Makefile and rename sbatch script

Add the L3 build/run targets and rename the sbatch script (so its name reflects what it actually runs now). After this task, `make all` builds three binaries (L0, L1, L2) — L3's source doesn't exist yet, so its build is conditional.

**Files:**
- Modify: `app/hip/Makefile`
- Rename + extend: `app/hip/run_l1_l2.sbatch` → `app/hip/run_staircase.sbatch`

- [ ] **Step 1: Replace `app/hip/Makefile` contents**

```makefile
HIPCC    ?= hipcc
# Auto-detect the local GPU arch via rocminfo; fall back to gfx942 (MI300X) if
# unavailable. Override explicitly with `make ARCH=gfx90a` for MI210, etc.
ARCH     ?= $(shell rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx/ {print $$2; exit}')
ifeq ($(strip $(ARCH)),)
ARCH := gfx942
endif
CXXFLAGS ?= -O2 -std=c++17 --offload-arch=$(ARCH) -D__AMDGCN_WAVEFRONT_SIZE=64

TARGETS := vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2 vdcores_hip_l3

all: $(TARGETS)

vdcores_hip_demo: vdcores_hip_demo.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

vdcores_hip_l1: vdcores_hip_l1.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

vdcores_hip_l2: vdcores_hip_l2.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

vdcores_hip_l3: vdcores_hip_l3.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

run-l0: vdcores_hip_demo
	./vdcores_hip_demo

run-l1: vdcores_hip_l1
	./vdcores_hip_l1

run-l2: vdcores_hip_l2
	./vdcores_hip_l2

run-l3: vdcores_hip_l3
	./vdcores_hip_l3

clean:
	rm -f $(TARGETS)

.PHONY: all run-l0 run-l1 run-l2 run-l3 clean
```

(Tabs in recipe lines.)

- [ ] **Step 2: `git mv` the sbatch script and update its loops**

```bash
cd /work1/maryhall/sameeran/work/vdcores
git mv app/hip/run_l1_l2.sbatch app/hip/run_staircase.sbatch
```

Then replace `app/hip/run_staircase.sbatch` contents with:

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
# REPO_DIR cascade: explicit override > SLURM submit dir > hardcoded default.
# The hardcoded default keeps the current setup working unchanged; the cascade
# lets another contributor or relocation point at a different checkout via
# `REPO_DIR=/path/to/repo sbatch app/hip/run_staircase.sbatch`.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/work1/maryhall/sameeran/work/vdcores}}"
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
for tgt in vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2 vdcores_hip_l3; do
  if [ -f "${tgt}.cpp" ]; then
    echo "--- build ${tgt} ---"
    make "${tgt}"
  else
    echo "--- skip ${tgt} (no source) ---"
  fi
done

for bin in vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2 vdcores_hip_l3; do
  if [ -x "./${bin}" ]; then
    echo "--- run ${bin} ---"
    timeout 10 "./${bin}" || echo "(${bin} exited non-zero)"
  fi
done

echo "=== job ${SLURM_JOB_ID} done ==="
date
```

- [ ] **Step 3: Local compile-check that L0/L1/L2 still build under the new Makefile**

(L3 source doesn't exist yet so `make all` will fail — that's expected. Build the three existing rungs only.)

```bash
cd /work1/maryhall/sameeran/work/vdcores/app/hip
make clean
make vdcores_hip_demo
make vdcores_hip_l1
make vdcores_hip_l2
ls -la vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2
```

Expected: three executables produced. No errors. **No GPU used** — this is a local compile.

- [ ] **Step 4: Commit**

```bash
cd /work1/maryhall/sameeran/work/vdcores
git add app/hip/Makefile app/hip/run_staircase.sbatch
git commit -m "$(cat <<'EOF'
hip-port: L3 — extend Makefile + rename sbatch to run_staircase.sbatch

Add vdcores_hip_l3 build target and run-l3 phony to the Makefile. The
SLURM script (renamed from run_l1_l2.sbatch since it now drives a
4-rung staircase) builds whichever sources exist and runs each binary
under a 10s timeout — same shape, expanded list. No semantic changes
to L0/L1/L2 build paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(`git mv` already staged the rename; the second sbatch update plus the Makefile change land in the same commit.)

---

### Task 2: Create `vdcores_hip_l3.cpp` as verbatim copy of L2

Same pattern as the L1→L2 copy step. Only the kernel name and print tag change. Compile-check locally; no GPU.

**Files:**
- Create: `app/hip/vdcores_hip_l3.cpp`

- [ ] **Step 1: Copy L2 → L3**

```bash
cd /work1/maryhall/sameeran/work/vdcores
cp app/hip/vdcores_hip_l2.cpp app/hip/vdcores_hip_l3.cpp
```

- [ ] **Step 2: Rename kernel `vdcores_l2_kernel` → `vdcores_l3_kernel`**

In `app/hip/vdcores_hip_l3.cpp`, find:

```cpp
void vdcores_l2_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
```

Replace with:

```cpp
void vdcores_l3_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
```

And find:

```cpp
  hipLaunchKernelGGL(vdcores_l2_kernel, grid, block, 0, stream, da, dc);
```

Replace with:

```cpp
  hipLaunchKernelGGL(vdcores_l3_kernel, grid, block, 0, stream, da, dc);
```

- [ ] **Step 3: Rename print tag `vdcores-hip-l2` → `vdcores-hip-l3`**

Find:

```cpp
  std::printf("[vdcores-hip-l2] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");
```

Replace with:

```cpp
  std::printf("[vdcores-hip-l3] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");
```

(Note: `2 * WAVE` is intentional here — Task 3 will change this to `3 * WAVE` along with the block-size change. T2 is verbatim copy + rename only.)

- [ ] **Step 4: Local compile-check**

```bash
cd /work1/maryhall/sameeran/work/vdcores/app/hip
make vdcores_hip_l3
```

Expected: `vdcores_hip_l3` executable produced. No GPU used.

- [ ] **Step 5: Confirm L3↔L2 diff is only the three substitutions**

```bash
diff app/hip/vdcores_hip_l2.cpp app/hip/vdcores_hip_l3.cpp
```

Expected: three lines change (kernel def, launch site, print tag). Nothing else.

- [ ] **Step 6: Commit**

```bash
cd /work1/maryhall/sameeran/work/vdcores
git add app/hip/vdcores_hip_l3.cpp
git commit -m "$(cat <<'EOF'
hip-port: add L3 source (verbatim copy of L2, renamed kernel + print tag)

vdcores_hip_l3.cpp is byte-identical to vdcores_hip_l2.cpp except for
the kernel name (vdcores_l3_kernel) and print tag ([vdcores-hip-l3]).
Confirms the build wiring before the 3-wave topology change. Compiled
locally; no GPU used yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Surgical change — split `mem` wave into `LD` (wave 0) + `ST` (wave 2)

The actual L3 architectural change. Bump block size; add `stored` signal; reshape the chunk loop to a 3-wave producer/consumer ring (LD → compute → ST → LD).

**Files:**
- Modify: `app/hip/vdcores_hip_l3.cpp`

- [ ] **Step 1: Bump `__launch_bounds__` to `3 * WAVE`**

In `app/hip/vdcores_hip_l3.cpp`, find:

```cpp
__global__ __launch_bounds__(2 * WAVE)
void vdcores_l3_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
```

Replace with:

```cpp
__global__ __launch_bounds__(3 * WAVE)
void vdcores_l3_kernel(const float* __restrict__ a,
                       float* __restrict__ c) {
```

- [ ] **Step 2: Add `stored` to the `__shared__` signal block + init it**

Find:

```cpp
  __shared__ LdsSignal loaded;     // mem-wave  → compute-wave
  __shared__ LdsSignal computed;   // compute-wave → mem-wave (store)
```

Replace with:

```cpp
  __shared__ LdsSignal loaded;     // LD wave  → compute wave
  __shared__ LdsSignal computed;   // compute  → ST wave
  __shared__ LdsSignal stored;     // ST       → LD wave (closes ring; gates slot reuse)
```

Find:

```cpp
  if (tid == 0) {
    loaded.counter   = 0;
    computed.counter = 0;
  }
  __syncthreads();
```

Replace with:

```cpp
  if (tid == 0) {
    loaded.counter   = 0;
    computed.counter = 0;
    stored.counter   = 0;
  }
  __syncthreads();
```

- [ ] **Step 3: Reshape the chunk loop into a 3-wave producer/consumer ring**

Find the entire chunk loop body (currently 2-wave). It looks like:

```cpp
  for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
    const int off = chunk * CHUNK;
    const unsigned target = static_cast<unsigned>(chunk + 1);

    // memory wavefront stages a global -> LDS via explicit async load sequence.
    // ...
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

Replace the entire chunk loop body with:

```cpp
  for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
    const int off = chunk * CHUNK;
    const unsigned target = static_cast<unsigned>(chunk + 1);

    // wave 0 (LD): wait for slot to be free (prev ST done), then async-load.
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
      arrive(loaded);                                    // signal: load done
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
```

Key topology changes vs L2:

- Wave 0 (was `mem`, did both load + store) now does **load only**, plus a new initial `wait_at_least(stored, chunk)` so it doesn't overwrite a slot still being drained.
- Wave 2 is **new** — handles the global-store half that L2's wave 0 used to handle, plus signals `stored` to release the slot.
- Wave 1 (compute) is unchanged in body; just sandwiched between LD and ST waves now.
- For chunk 0 the LD wave's `wait_at_least(stored, 0)` is satisfied immediately (counter is 0, target is 0).

- [ ] **Step 4: Update the host-side launch — block size and the print's `threads/block` argument**

Find:

```cpp
  dim3 grid(N_BLOCKS), block(2 * WAVE);
  hipLaunchKernelGGL(vdcores_l3_kernel, grid, block, 0, stream, da, dc);
```

Replace with:

```cpp
  dim3 grid(N_BLOCKS), block(3 * WAVE);
  hipLaunchKernelGGL(vdcores_l3_kernel, grid, block, 0, stream, da, dc);
```

Find:

```cpp
  std::printf("[vdcores-hip-l3] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 2 * WAVE, errors, errors ? "FAIL" : "PASS");
```

Replace with:

```cpp
  std::printf("[vdcores-hip-l3] N=%d blocks=%d threads/block=%d  errors=%d  %s\n",
              N, N_BLOCKS, 3 * WAVE, errors, errors ? "FAIL" : "PASS");
```

- [ ] **Step 5: Local compile-check**

```bash
cd /work1/maryhall/sameeran/work/vdcores/app/hip
make vdcores_hip_l3
```

Expected: clean build. No GPU used.

If hipcc rejects `__launch_bounds__(3 * WAVE)` (192 threads/CTA) for any reason, STOP — that's a real architectural blocker. 192 is well within MI300X limits, so this should not happen.

- [ ] **Step 6: Confirm the L3↔L2 diff is bounded to the topology change**

```bash
diff app/hip/vdcores_hip_l2.cpp app/hip/vdcores_hip_l3.cpp
```

Expected hunks: launch_bounds, kernel rename, `__shared__` signal block addition, init block addition, chunk loop body rewrite, `block(3*WAVE)`, `threads/block` printf arg, print tag. No other changes.

- [ ] **Step 7: Commit**

```bash
cd /work1/maryhall/sameeran/work/vdcores
git add app/hip/vdcores_hip_l3.cpp
git commit -m "$(cat <<'EOF'
hip-port: L3 — split mem wave into LD (wave 0) + ST (wave 2) waves

Bump block size from 2*WAVE to 3*WAVE (192 threads/CTA, three
wavefronts on AMD wave64). Add a third LdsSignal "stored" that
ST-wave bumps after each global drain so the LD wave can reuse the
single LDS slot for the next chunk. Wave 0 now does load only; new
wave 2 takes over the global-store half. Compute wave (wave 1) is
unchanged in body. Workload is still pure copy c=a; buffering is
still single-slot. Double-buffering arrives at L4 as a separate spec.

Validates that 3 wavefronts can cooperate in one CTA on AMD wave64
without the alloc+ST shared-wavefront hazard that broke the
megakernel port. Compiled locally on the login node; verified by
sbatch in the final task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Append L3 paragraph to `app/hip/README.md`

Document the new topology and the `stored` signal.

**Files:**
- Modify: `app/hip/README.md`

- [ ] **Step 1: Append after the existing L2 section, before the SLURM-submission section**

In `app/hip/README.md`, find the heading `### Running all three on MI300X via SLURM` (or whatever the existing run-all section is called).

Insert immediately **before** that heading:

```markdown
### L3 — `vdcores_hip_l3`

Same load surface and queue primitive as L2, but the `mem` wave is split: wave 0 is now LD-only, wave 2 is the new ST wave, and a third `LdsSignal` (`stored`) closes the ring so wave 0 can't overwrite a slot that wave 2 is still draining. Block size is now `3 * WAVE = 192` threads per CTA.

Validates that 3 wavefronts cooperate cleanly in one CTA on AMD wave64 — i.e. the alloc+ST shared-wavefront hazard from the megakernel port is structurally absent here. Workload still `c = a`; still single buffer (overlap arrives at L4).

```bash
make run-l3
```

Expected: `[vdcores-hip-l3] N=32768 blocks=16 threads/block=192  errors=0  PASS`

```

Then update the SLURM-submission section to reference `run_staircase.sbatch` (instead of `run_l1_l2.sbatch`) and to mention four PASS lines instead of three. Find:

```markdown
### Running all three on MI300X via SLURM

```bash
sbatch run_l1_l2.sbatch
```

The job builds whichever sources are present and runs each binary under a 10s timeout. All three lines should print `errors=0 PASS`.
```

Replace with:

```markdown
### Running the staircase on MI300X via SLURM

```bash
sbatch run_staircase.sbatch
```

The job builds whichever sources are present and runs each binary under a 10s timeout. All four (L0/L1/L2/L3) lines should print `errors=0 PASS`.
```

- [ ] **Step 2: Commit**

```bash
cd /work1/maryhall/sameeran/work/vdcores
git add app/hip/README.md
git commit -m "$(cat <<'EOF'
hip-port: README — document L3 (3-wave LD/compute/ST staircase rung)

Add an L3 paragraph between L2 and the SLURM section explaining the
3-wave topology, the new "stored" signal, and the expected PASS line
(threads/block=192). Update the SLURM section to reference
run_staircase.sbatch and to mention four PASS lines.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Single end-to-end sbatch verification + tag

The one and only GPU run for L3.

- [ ] **Step 1: Confirm L0/L1/L2 sources are byte-identical to `vdcores-amd-port-l1-l2-v1`**

```bash
cd /work1/maryhall/sameeran/work/vdcores
git diff vdcores-amd-port-l1-l2-v1 -- app/hip/vdcores_hip_demo.cpp app/hip/vdcores_hip_l1.cpp app/hip/vdcores_hip_l2.cpp
```

Expected: empty diff. Three baseline rungs have not been touched.

- [ ] **Step 2: Submit the staircase sbatch**

```bash
cd /work1/maryhall/sameeran/work/vdcores
sbatch app/hip/run_staircase.sbatch
```

Capture the job ID. Wait for `COMPLETED` (sacct poll, ~1 minute).

- [ ] **Step 3: Verify all four rungs PASS**

```bash
ls -t /work1/maryhall/sameeran/work/vdcores/vdcores-hip-*.out | head -1 | xargs cat | grep -E "vdcores-hip(-l[123])?\] N="
```

Expected (four lines):

```
[vdcores-hip] N=32768 blocks=16 threads/block=128  errors=0  PASS
[vdcores-hip-l1] N=32768 blocks=16 threads/block=128  errors=0  PASS
[vdcores-hip-l2] N=32768 blocks=16 threads/block=128  errors=0  PASS
[vdcores-hip-l3] N=32768 blocks=16 threads/block=192  errors=0  PASS
```

(L3's `threads/block` is `192` — the 3-wave kernel — while L0/L1/L2 stay at `128`.)

If L3 hangs (timeout 10 fires, `(vdcores_hip_l3 exited non-zero)` printed), the suspect is the `stored` signal ring: check the off-by-one in `wait_at_least(stored, chunk)` for chunk 0 (counter must start at 0, target must be 0, so the wait passes immediately).

If L3 prints `FAIL`, the topology change has a real correctness bug. Do NOT tag. Investigate the diff against L2 (only the topology change is in scope).

- [ ] **Step 4: Tag the L3 milestone**

```bash
cd /work1/maryhall/sameeran/work/vdcores
git tag -a vdcores-amd-port-l3-v1 -m "VDCores AMD port: L3 (3-wave LD/compute/ST) verified PASS on MI300X"
```

- [ ] **Step 5: Final report**

Print the four PASS lines, the tag, and the commit chain:

```bash
git log --oneline vdcores-amd-port-l1-l2-v1..HEAD
```

---

## Self-review

- **Spec coverage:** every spec section maps to a task. Architecture (T1+T3), components (T2+T3), data flow (T3), error handling (T3+T5), testing (T5), README (T4).
- **GPU usage:** zero between T1 and T4 (all local hipcc compile-checks). One sbatch in T5. Matches the user directive.
- **Bisection property:** L3↔L2 diff is bounded to the 3-wave topology change. If L3 regresses but L2 still passes, the diff *is* the suspect.
- **No placeholders:** every step has actual code and exact commands.
- **Type/name consistency:** `vdcores_l3_kernel`, `[vdcores-hip-l3]`, `stored` (LdsSignal) are used consistently across definition, launch, and waits.
