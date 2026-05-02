# VDCores HIP demo

Minimal proof-of-concept showing the VDCores decoupled memory/compute model on AMD HIP.
One CTA = one **memory wavefront** + one **compute wavefront**, communicating through LDS.

Workload: `c = a + b` over chunks staged into LDS.

## Files
- `vdcores_hip_demo.cpp` — kernel + host driver (HIP)
- `Makefile` — builds with `hipcc`, auto-detecting `--offload-arch` via `rocminfo`
  (falls back to `gfx942` / MI300X). Override with `make ARCH=gfx90a` for MI210.

## Build & run
```bash
module load rocm        # or whatever your env uses
make                    # auto-detects local GPU arch
make run
```
Verified: PASS on **MI210 (gfx90a)** and **MI300X (gfx942)**.

Expected output:
```
[vdcores-hip] N=32768 blocks=16 threads/block=128  errors=0  PASS
```

## What this maps from the VDCores model

| VDCores concept (CUDA/Hopper) | This demo (HIP/MI300X)            |
|-------------------------------|------------------------------------|
| Warp = 32 threads             | Wavefront = 64 threads             |
| Memory core (TMA loads)       | Wave 0 staging global → LDS        |
| Compute core (WGMMA / ALU)    | Wave 1 doing `+` on LDS data       |
| Slot/queue between cores      | LDS arrays + `__syncthreads()`     |
| `cudaStream_t` / `cudaEvent_t`| `hipStream_t` (non-blocking)       |
| `cudaMemcpyAsync`             | `hipMemcpyAsync`                   |
| `cudaLaunchKernel` (`<<< >>>`)| `hipLaunchKernelGGL`               |
| `sm_90a`                      | `gfx90a` / `gfx942` via `--offload-arch` |

## What's intentionally NOT here (per scope)

- No async global→LDS intrinsic (`__builtin_amdgcn_global_load_lds`) — plain LDS staging
- No double-buffering / pipelining
- No GEMM / attention / RoPE — just vector add
- No Python / PyTorch wrapper — pure C++
- No multi-block scheduling or queue infrastructure

This is a *one-kernel* sanity check that the VDCores partitioning idea (split a CTA
into memory-only and compute-only wavefronts, sync via LDS) compiles and runs on
ROCm/HIP. Extend from here.

## Staircase rungs

The directory now contains a staircase of three binaries, each one surgical change from the rung below. The staircase is the debugger: if a rung regresses, the diff against the rung below it is the suspect list.

### L0 — `vdcores_hip_demo`

Already documented above. Plain LDS staging + `__syncthreads`. Workload `c = a + b`.

### L1 — `vdcores_hip_l1`

Same skeleton as L0, but wave 0's global→LDS staging uses an explicit two-phase async sequence:

```cpp
// Phase 1: post N flat_load_dword loads into VGPRs (in flight simultaneously).
asm volatile("flat_load_dword %0, %1 sc0 sc1" : "=v"(tmp) : "v"(...));
// Phase 2: drain the VMEM pipeline.
asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
// Phase 3: ds_write_b32 each VGPR into its LDS slot.
asm volatile("ds_write_b32 %0, %1" :: "v"(lds_off), "v"(tmp));
asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
```

Note: the original goal was an async-direct global→LDS path via `__builtin_amdgcn_global_load_lds`. That intrinsic is unavailable on ROCm 7.2 / clang-22 (LLVM "Cannot select" backend crash) and the spec's `s_mov_b32 m0` + `global_load_lds_dword` fallback hit an SGPR/VGPR constraint mismatch. L1 instead validates the AMD memory-model dialect we'll use everywhere downstream — explicit `s_waitcnt` discipline + inline-asm comfort. Revisit the async-direct path when the toolchain catches up.

Workload simplified to `c = a` (pure copy — no compute work). Cross-wave sync still uses `__syncthreads`. Build + run:

```bash
make run-l1
```

Expected: `[vdcores-hip-l1] N=32768 blocks=16 threads/block=128  errors=0  PASS`

### L2 — `vdcores_hip_l2`

Same load surface as L1. The three cross-wave `__syncthreads` calls are replaced with an LDS-counter `arrive(s) / wait_at_least(s, target)` queue. Two signals per CTA: `loaded` (mem→compute) and `computed` (compute→mem-store). Single buffer, monotonic counters, fence-before-atomic on the producer side. Build + run:

```bash
make run-l2
```

Expected: `[vdcores-hip-l2] N=32768 blocks=16 threads/block=128  errors=0  PASS`

### Running all three on MI300X via SLURM

```bash
sbatch run_l1_l2.sbatch
```

The job builds whichever sources are present and runs each binary under a 10s timeout. All three lines should print `errors=0 PASS`.
