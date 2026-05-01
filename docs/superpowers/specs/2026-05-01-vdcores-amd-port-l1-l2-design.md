# VDCores AMD port — L1+L2 design

**Date:** 2026-05-01
**Branch:** `hip-port` (this spec lives on the same branch but does not depend on it)
**Scope:** v1 of an AMD MI300X port of VDCores. **L1 (async global→LDS load) and L2 (LDS-counter queue) only.** L3+ are out of scope and will get their own specs.

## Problem

The current `hip-port` branch is a mechanical hipify port of the VDCores CUDA megakernel. It compiles on MI300X (gfx942, wave64) but `dae_copy_smoke.py` deadlocks on the simplest possible workload — 1 SM, single 1024-byte `TmaLoad1D + TmaStore1D` — and has resisted printf-based debugging for a day. Hopper-specific primitives (TMA `cp.async.bulk.tensor`, mbarrier with TX-counts, WGMMA, warpgroup register tiering) are stubbed with `__builtin_trap`, so even after the deadlock is fixed the hot path traps on real workloads.

The port is fighting the hardware on two fronts at once: the *pipeline harness* (alloc/LD/ST/compute warps with shared-memory queues) and the *primitives* the harness assumes (TMA, mbarrier, WGMMA). We can't bisect a failure between the two.

## Goal

Build the architectural ideas of VDCores — decoupled memory/compute waves, LDS slot handoff, eventually instruction-stream interpretation — on AMD-native primitives, **bottom-up from a known-good baseline.** Each rung is a runnable, verifiable binary. The staircase *is* the debugger.

`app/hip/vdcores_hip_demo.cpp` already PASSes on MI300X with `c = a + b` over LDS — that's L0. This spec defines the next two rungs.

| Rung | Adds | Validates | Status |
|---|---|---|---|
| L0 | mem-wave + compute-wave + `__syncthreads` LDS handoff | DAE wave partitioning idea on AMD | ✅ already PASS on MI300X |
| **L1** | `__builtin_amdgcn_global_load_lds` + `s_waitcnt vmcnt(0)` | AMD's TMA-equivalent async load primitive, in isolation | this spec |
| **L2** | LDS-counter `arrive(N) / wait_at_least(target)` queue | Replacement for `cuda::barrier` / mbarrier, in isolation | this spec |
| L3 | 3rd wave (alloc → LD → compute → ST), double-buffer | Multi-stage pipelining; mirrors DAE topology | future spec |
| L4 | Minimal `MInst` interpreter loop | Instruction-stream model | future spec |
| L5 | Wire into Python `Launcher`, target `dae_copy_smoke.py` | Python ↔ HIP path | future spec |
| L6 | MFMA-backed GEMM as compute task | WGMMA replacement | future spec |

**v1 is done when L1 and L2 each print `errors=0 PASS` on MI300X.**

## Locked design decisions

| | |
|---|---|
| Wave width | wave64 (matches L0 baseline; wave32 vs wave64 is a multi-wave-divergence concern that arrives at L3) |
| Workload | Pure copy `c = a` (single memcmp verification) |
| Buffering | Single buffer (one LDS slot for `lds_a`, one for `lds_c`) |
| File layout | New files `vdcores_hip_l1.cpp`, `vdcores_hip_l2.cpp` in `app/hip/`. L0 untouched. |
| Build approach | Bottom-up staircase. Each rung is a separate binary. `make run-l0`/`run-l1`/`run-l2`. |
| Sync mechanism | L1 keeps `__syncthreads` (only the load primitive changes); L2 swaps `__syncthreads` for LDS-counter signals (only the sync changes). |
| No shared header | L1 and L2 each inline their primitives. Extract to `app/hip/include/` only when L3 needs it (YAGNI). |

## Architecture

Two new standalone HIP binaries, each a copy of the rung below with one surgical change.

```
app/hip/
├── vdcores_hip_demo.cpp     [L0 — UNCHANGED]
├── vdcores_hip_l1.cpp       [L1 — NEW]
├── vdcores_hip_l2.cpp       [L2 — NEW]
├── Makefile                 [updated: run-l0 / run-l1 / run-l2 targets, all builds, all-clean]
└── README.md                [updated: one paragraph per rung]
```

Per-binary topology, identical to L0:

```
1 CTA = 128 threads = 2 wavefronts (wave64 each)
  wave 0 = "memory wave"   (global ↔ LDS)
  wave 1 = "compute wave"  (LDS → LDS, trivial passthrough since workload is pure copy)

Workload: c = a, processed in 8 chunks of 256 floats per chunk.
LDS:      lds_a[256], lds_c[256]  (lds_b dropped — no `+ b`)
Grid:     16 blocks × 1024 floats/block = 16384 floats total. Same N as L0.
```

Per-chunk data flow has two handoff points → two queue signals:

```
[wave 0]  global a → lds_a   ──signal-1→
                                          [wave 1]  lds_a → lds_c   ──signal-2→
                                                                                 [wave 0]  lds_c → global c
```

L1 uses `__syncthreads()` for both signals. L2 swaps each `__syncthreads()` for an LDS-counter `arrive(N)` / `wait_at_least(target)` pair.

## Components

### L1 — `vdcores_hip_l1.cpp` (~150 lines)

Same skeleton as L0; **only the load instruction changes**.

The async-load primitive (gfx942 native):

```cpp
// Each lane issues one async global→LDS dword (no register round-trip).
__builtin_amdgcn_global_load_lds(
    /*global src*/  &a[base + off + lane],
    /*lds   dst*/   &lds_a[lane],
    /*size  bytes*/ 4,
    /*offset*/      0,
    /*aux*/         0);
// ...one issue per CHUNK/64 inner-loop iterations across all 64 lanes...
asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
```

Issued by all 64 lanes of wave 0; one wait gates all in-flight loads for the chunk. `s_waitcnt vmcnt(0)` is the targeted wait — the alternative `__builtin_amdgcn_s_waitcnt(0)` waits on all counters and is overkill here.

Synchronisation between waves stays as `__syncthreads()` — unchanged from L0. The point of L1 is to validate the new load primitive in isolation, with the simplest possible signal.

### L2 — `vdcores_hip_l2.cpp` (~170 lines)

Starts as a copy of L1; **only the synchronisation mechanism changes**.

The LDS-counter queue primitive, defined inline at top of the file:

```cpp
struct LdsSignal {           // single LDS word
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

Two `LdsSignal`s per CTA: `loaded` (mem→compute) and `computed` (compute→mem-store). Each chunk increments each signal exactly once. Inside chunk-loop iteration `k`:

- wave 0 issues async load → `s_waitcnt vmcnt(0)` → `arrive(loaded)`
- wave 1 `wait_at_least(loaded, k+1)` → `lds_c[i] = lds_a[i]` → `arrive(computed)`
- wave 0 `wait_at_least(computed, k+1)` → store

Single buffer, monotonic counters, no ring. As simple as the protocol gets.

`atomicAdd(p, 0u)` for the spin-poll matches the pattern we already validated in `include/dae/hip_compat.cuh:244`: a plain or volatile load can be hoisted past `s_sleep` by the AMD backend; an atomic op is a hard memory barrier that forces a fresh LDS fetch every iteration.

### Makefile changes

```makefile
TARGETS := vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2

all: $(TARGETS)

vdcores_hip_demo: vdcores_hip_demo.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<
vdcores_hip_l1:   vdcores_hip_l1.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<
vdcores_hip_l2:   vdcores_hip_l2.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

run-l0: vdcores_hip_demo ; ./vdcores_hip_demo
run-l1: vdcores_hip_l1   ; ./vdcores_hip_l1
run-l2: vdcores_hip_l2   ; ./vdcores_hip_l2

clean:
	rm -f $(TARGETS)
```

### README changes

One paragraph per rung explaining what's new vs the rung below and what `PASS` output to expect. Existing L0 paragraph stays as-is.

## Per-chunk data flow (detail)

```
chunk loop k ∈ [0, 8):

  wave 0:
    for i = lane; i < CHUNK; i += WAVE:
      __builtin_amdgcn_global_load_lds(&a[base+off+i], &lds_a[i], 4, 0, 0)
    s_waitcnt vmcnt(0)
    arrive(loaded)                                       ──signal──┐
                                                                    ▼
  wave 1:
    wait_at_least(loaded, k+1)
    for i = lane; i < CHUNK; i += WAVE:
      lds_c[i] = lds_a[i]
    arrive(computed)                                     ──signal──┐
                                                                    ▼
  wave 0:
    wait_at_least(computed, k+1)
    for i = lane; i < CHUNK; i += WAVE:
      c[base+off+i] = lds_c[i]
```

After the loop, `loaded.counter == computed.counter == 8`.

## Error handling

Three failure modes, each surfaces as hard PASS/FAIL on the host:

1. **Build fails** (e.g., `__builtin_amdgcn_global_load_lds` signature differs from what we wrote against ROCm 7.2). Hipcc error halts the build. No silent fallback.
2. **Correctness mismatch** — host loop diffs `c[i]` against `a[i]`, prints `errors=N FAIL` and `exit(N)`. Same shape as L0's PASS/FAIL print.
3. **Hang** — sbatch wraps `timeout 30` around `make run-l1` / `make run-l2`. Kernel deadlock kills cleanly with exit 124 and the job log shows up to the last flushed printf. Same pattern as the existing `run_batch.sbatch`.

No try/catch, no fallbacks, no recovery paths. This is a smoke test — failures must be loud.

## Testing

Each rung passes when its binary prints `errors=0 PASS` on `make run-l<N>`.

- **L0** baseline (already verified): `[vdcores-hip] N=32768 blocks=16 threads/block=128 errors=0 PASS`
- **L1** target: same shape, modulo binary name. `errors=0` is the only success criterion.
- **L2** target: same.

No throughput numbers in v1 — perf is a separate concern that arrives with L3 (multi-stage buffering). We'll add `hipEventElapsedTime` then.

Verification ladder for v1 done:

```
make clean && make all && make run-l0 && make run-l1 && make run-l2
```

All three must print `PASS`. If L1 regresses but L0 still passes, the diff between L1 and L0 is the suspect — the load primitive. If L2 regresses but L1 still passes, the diff between L2 and L1 is the suspect — the queue primitive. The staircase makes bisection trivial.

## Out of scope (explicit YAGNI)

- Throughput / latency measurements
- Double-buffering or N-slot ring
- More than 2 wavefronts per CTA
- Wave32 mode
- MFMA / GEMM / any compute beyond `lds_c[i] = lds_a[i]`
- TMA descriptors (`cuTensorMapEncodeTiled` and friends)
- Python launcher integration
- Touching the existing megakernel branch (`include/dae/`, `src/runtime.cu`, etc.)
- Diagnostic instrumentation beyond the L0-style `errors=N` print

Each of these is fine on its own merits; none belong in v1.

## Risks and assumptions

1. **`__builtin_amdgcn_global_load_lds` signature.** The exact arg order/count varies across LLVM versions. ROCm 7.2 / clang-22 is what the cluster has — we'll verify against the headers during impl. If it differs from the spec, we adjust the L1 source (one site to fix). Fallback: inline-asm `global_load_lds_dword` directly. Either is acceptable on gfx942.
2. **`s_waitcnt vmcnt(0)` syntax.** Inline asm string is stable on gfx942 but document a fallback to `__builtin_amdgcn_s_waitcnt(0)` (overkill but portable).
3. **LDS-atomic visibility across waves.** `atomicAdd` on LDS is well-defined within a CTA on CDNA3; `__threadfence_block` provides the necessary release/acquire ordering. We've already used this pattern in `hip_compat.cuh` and it survives the LLVM optimiser.
4. **The L2 spin-poll is busy-wait.** That's fine for a smoke test (one CTA, two waves). Becomes a perf concern at L3+.
5. **GPU-hour budget.** Each L1/L2 iteration is a ~30s sbatch (build + 1s run + teardown). Twenty iterations to land L1+L2 is ≈10 GPU-minutes — well under budget.

## Open questions

None at spec time. All decisions are locked above. Implementation can begin.
