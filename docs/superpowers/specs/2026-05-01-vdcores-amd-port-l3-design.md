# VDCores AMD port — L3 design (3rd wave: LD + compute + ST)

**Date:** 2026-05-01
**Branch:** `hip-port`
**Scope:** L3 only — split the L2 `mem` wave into separate `LD` (load) and `ST` (store) waves. **Still single-buffer.** Double-buffering arrives at L4 as a separate spec.
**Builds on:** `docs/superpowers/specs/2026-05-01-vdcores-amd-port-l1-l2-design.md` (L1+L2 done, tagged `vdcores-amd-port-l1-l2-v1`)

## Why split the staircase here

The original L1+L2 spec gestured at "3rd wave + double-buffer" as one rung. That's two surgical changes — splitting them keeps the bisection property: if L3 regresses, the diff against L2 is *only* the topology change; if L4 regresses, the diff against L3 is *only* the buffering change.

## Goal

Validate that **3 wavefronts can cooperate in one CTA on AMD wave64** without the alloc+ST shared-wavefront hazard that broke the original megakernel port. Replace L2's single `mem` wave with two distinct waves — one for global→LDS loads, one for LDS→global stores — with the compute wave between them. Workload unchanged: `c = a`.

| Rung | Adds | Validates | Status |
|---|---|---|---|
| L2 | LDS-counter `arrive`/`wait_at_least` queue | Replacement for `cuda::barrier`/mbarrier | ✅ tagged `vdcores-amd-port-l1-l2-v1` |
| **L3** | 3rd wavefront; mem wave split into LD + ST; new `stored` signal closes the producer/consumer ring | 3-wave divergence safety on AMD wave64; the LD/compute/ST topology of DAE | this spec |
| L4 | Double-buffered LDS slots (ping-pong) | Multi-stage pipelining; overlap LD with compute and ST | future spec |
| L5 | Minimal `MInst` interpreter loop | Instruction-stream model | future spec |
| L6 | Wire into Python `Launcher` (`dae_copy_smoke.py` PASS) | Python ↔ HIP path | future spec |
| L7 | MFMA-backed compute task (GEMM/GEMV) | WGMMA replacement | future spec |

**v1.1 is done when L3 prints `errors=0 PASS` on MI300X (and L0/L1/L2 still PASS).**

## Locked design decisions

| | |
|---|---|
| Wave width | wave64 (matches L0–L2) |
| Wave count | **3** (was 2): wave 0 = LD, wave 1 = compute, wave 2 = ST |
| Threads/CTA | **192** (was 128) — `__launch_bounds__(3 * WAVE)` |
| Workload | Pure copy `c = a` (unchanged from L1/L2) |
| Buffering | **Single buffer** (one `lds_a`, one `lds_c` — same as L2). Double-buffer is L4. |
| Signals | **Three** `LdsSignal` per CTA: `loaded` (LD→compute), `computed` (compute→ST), `stored` (ST→LD, closes the ring) |
| File layout | New file `vdcores_hip_l3.cpp` based on L2. L0/L1/L2 untouched. |
| Sync mechanism | Same `arrive` / `wait_at_least` primitive as L2 — no header changes |
| GPU budget | Compile-check on login node (no GPU). ONE sbatch at the very end to verify all four rungs PASS in one shot. |

## Architecture

```
1 CTA = 192 threads = 3 wavefronts (wave64 each):
  wave 0 = LD       (global a → lds_a)
  wave 1 = compute  (lds_a → lds_c, trivial passthrough for c=a)
  wave 2 = ST       (lds_c → global c)

Single LDS slot pair (lds_a, lds_c). Three monotonic signals.
N_CHUNKS = 8 chunks of CHUNK = 256 floats each. Same workload size as L2.
```

## Per-chunk data flow

For chunk `k ∈ [0, 8)`, target `t = k + 1`:

```
[wave 0 LD]   wait_at_least(stored,   k)         ← gate on slot being free (prev ST done)
              flat_load_dword + s_waitcnt vmcnt(0)
              ds_write_b32   + s_waitcnt lgkmcnt(0)
              arrive(loaded)                                ──signal──┐
                                                                       ▼
[wave 1 cmp]  wait_at_least(loaded,   t)
              lds_c[i] = lds_a[i]                                      
              arrive(computed)                              ──signal──┐
                                                                       ▼
[wave 2 ST]   wait_at_least(computed, t)
              c[base+off+i] = lds_c[i]
              arrive(stored)                                ──signal──┐
                                                                       ▼
                                                          (next iter LD waits here)
```

For chunk 0 the LD wave's `wait_at_least(stored, 0)` is satisfied immediately (counter starts at 0, target is 0). For chunk 1+, LD must wait for the previous chunk's ST to complete — that's what makes this single-buffered: at most one chunk in-flight.

After the loop, all three counters equal `N_CHUNKS = 8`.

## Components

### `vdcores_hip_l3.cpp` (≈190 lines)

Starts as a copy of `vdcores_hip_l2.cpp`. Surgical changes:

1. **Block size**: `dim3 block(2 * WAVE)` → `dim3 block(3 * WAVE)` and `__launch_bounds__(2 * WAVE)` → `__launch_bounds__(3 * WAVE)`.
2. **Add `stored` signal**: `__shared__ LdsSignal stored;` next to `loaded` and `computed`. Init to 0 in the same `if (tid == 0) { ... }` block.
3. **Wave-0 (was `mem` doing both load + store) becomes LD-only**: the ST half of L2's wave-0 block moves out to a new `if (wave == 2)` branch. Wave 0 also waits on `stored, k` at the **start** of each iteration so the slot is free before reusing.
4. **New wave-2 (ST) block**: structurally mirrors wave-0's old store half. Waits `computed, t`, drains LDS→global, arrives `stored`.
5. **Kernel rename** `vdcores_l2_kernel` → `vdcores_l3_kernel`. Print tag `[vdcores-hip-l2]` → `[vdcores-hip-l3]`.

The `LdsSignal` struct, `arrive`, and `wait_at_least` definitions are copied verbatim from L2 (no shared header — same YAGNI as L1/L2).

### `app/hip/Makefile`
Add `vdcores_hip_l3` build target and `run-l3` phony. Add to `TARGETS`. `clean` already iterates over `$(TARGETS)`.

### `app/hip/run_l1_l2.sbatch`
Rename to `run_l1_l2_l3.sbatch` (or just extend the existing one — see plan). Update the build and run loops to include `vdcores_hip_l3`. Same 10s timeout per binary, same `REPO_DIR` cascade.

### `app/hip/README.md`
Append L3 paragraph after the existing L2 section. Document the 3-wave topology and the new `stored` signal.

## Error handling

Same model as L1/L2: each binary contains its own host-side correctness check (`errors=N`, prints `PASS`/`FAIL`). Three failure modes:

1. **Build fails** (e.g., `__launch_bounds__(192)` rejected) — hipcc halts; no silent fallback. Compile-check locally on login node before any sbatch.
2. **Correctness mismatch** — host loop diffs `c[i]` against `a[i]`, prints `errors=N FAIL`, `exit(N)`.
3. **Hang** — the most likely L3-specific risk: the `stored` signal forms a producer/consumer **ring** (LD → compute → ST → LD). If any wave's wait condition is wrong (e.g., off-by-one in target/counter), all three waves deadlock. The 10s sbatch `timeout` kills cleanly with exit 124. Diagnose by inspecting the diff against L2 — only the topology change is in scope.

## Testing

Each rung still passes when its binary prints `errors=0 PASS`. v1.1 verification ladder:

```
make clean && make all && make run-l0 && make run-l1 && make run-l2 && make run-l3
```

In sbatch form: a single submitted job builds whichever sources exist and runs each binary; we grep for all four PASS lines.

To minimize GPU usage: **all compile-checks run on the login node first** (`hipcc` is available there at `/opt/rocm-7.2.0/bin/hipcc`; compilation needs no GPU). Sbatch is submitted once, at the very end, after the full 4-rung build is green locally.

## Risks and assumptions

1. **3-wave divergence on wave64.** With 3 wavefronts (each its own physical 64-thread unit), each `if (wave == N)` branch executes under that wavefront's own EXEC mask without inter-wavefront divergence hazards. This is the *opposite* of the megakernel deadlock (where alloc+ST shared a wavefront). Should be safe; will be verified by the sbatch.
2. **`__launch_bounds__(192)` on gfx942.** 192 threads/CTA is well under MI300X's 1024-thread limit and well under the 1536-thread occupancy preferred. Should be fine.
3. **LDS budget unchanged.** `lds_a[256] + lds_c[256] + 3*LdsSignal = 256*4 + 256*4 + 12 = 2060 bytes` — far below MI300X's 64 KiB/CU LDS.
4. **The `stored` signal ring at chunk 0.** With `stored.counter == 0` initially and `wait_at_least(stored, 0)`, lane-0 reads `0 < 0 == false` → exits immediately. Off-by-one risk localized to this initial condition; verifiable by tracing chunk-0 behavior on paper.
5. **GPU-hour budget.** Per the user's directive, this rung uses ZERO GPU until the very last verification sbatch — all iteration happens on the login node via local hipcc compilation.

## Out of scope (explicit YAGNI)

- Double-buffering / pipelining (L4)
- More than 3 waves
- Wave32 mode
- MFMA / GEMM / any non-copy compute
- Python launcher integration
- Throughput / latency measurement
- Touching the megakernel branch (`include/dae/`, `src/runtime.cu`)
- Touching `app/hip/vdcores_hip_demo.cpp`, `vdcores_hip_l1.cpp`, or `vdcores_hip_l2.cpp` (those are tagged baselines)

## Open questions

None at spec time. Implementation begins with a local compile-check.
