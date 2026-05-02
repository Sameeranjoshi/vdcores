# Path B spike — findings (2026-05-01)

## TL;DR

**Root cause of the original `dae_copy_smoke.py` deadlock confirmed:** the mem group's 4 logical warps (alloc, ST, LD0, LD1) are pinned to two physical wave64 wavefronts because `numThreadsPerWarp=32` is hardcoded throughout the codebase. Wave-level `if/else` divergence serializes alloc and ST — alloc blocks on a barrier ST is supposed to decrement, but ST never gets the EXEC mask because alloc hasn't returned. Classic two-warps-one-wavefront deadlock.

**Fix is real but scope is multi-day**, not the 1-hour spike we time-boxed. Going Path A (continue staircase to L5+L6).

## Symptoms (recap)

From `vdcores-302828.out` and similar:

```
[0][Compute] [Copy][i=0] before wait        ← compute waiting on c2m (slot to come back filled)
[0][LDU] [LD Warp][AMD] Start LD warp execution   ← LD blocked at first m2ld.wait()
[0][PRE-DISPATCH] thread 128 ... warp_id=0  ← alloc reached dispatch
                                            ← alloc never prints PRE-ALLOC-CALL or CFU-PROBE-A
```

Truncation point: just before alloc enters the `if (warp_id == 0)` branch. We hypothesized two causes:

1. ~~ST traps on a TMA-store stub, killing its shared wavefront with alloc.~~ **Disproven.** `include/dae/pipeline/stwarp.cuh:32-54` has a working AMD branch for `OP_ALLOC_WB_TMA_STORE_1D` — synchronous LDS→global via uint4 wide stores. Only multi-D TMA stores trap.
2. ~~`cuda::barrier` shim's mbarrier emulation has a hoist bug.~~ **Already patched** twice (volatile load → atomicAdd-zero spin) and confirmed working by the L2 staircase rung.

The actual root cause is structural:

## The real bug

`include/dae/context.cuh:37`:
```cpp
static constexpr int numThreadsPerWarp = 32;
static constexpr int numThreads = numThreadsPerWarp * (numComputeWarps + numMemoryWarps);  // 256
```

`include/dae/dae2.cuh:38`:
```cpp
int warp_id = (thread_id % 128) / 32;        // CUDA warp = 32 threads
```

`include/dae/dae2.cuh:166-200`:
```cpp
if (warp_id == 0)      { allocwarp_execute(...); }       // threads 128-159
else if (warp_id == 1) { stwarp_execute_singlethread(...); }  // threads 160-191
else if (warp_id >= 2) { ldwarp_execute_singlethread(...); }  // threads 192-255
```

On AMD MI300X (gfx942 / CDNA3), the wavefront size is **fixed at 64** — gfx942 does not support wave32 (verified per AMD ISA reference). So the actual wavefront layout is:

| Wavefront | Thread range | Logical warps inside |
|---|---|---|
| WF 0 | 0–63 | compute warps 0+1 |
| WF 1 | 64–127 | compute warps 2+3 |
| **WF 2** | **128–191** | **alloc (warp_id=0) + ST (warp_id=1)** ← shared |
| **WF 3** | **192–255** | **LD0 (warp_id=2) + LD1 (warp_id=3)** ← shared |

When WF 2 hits the `if (warp_id == 0) … else if (warp_id == 1) …` chain, AMDGCN serializes the branches under EXEC mask:
1. Wave runs `allocwarp_execute()` body with EXEC = lanes 128–159 active.
2. Wave then runs `stwarp_execute_singlethread()` body with EXEC = lanes 160–191 active.

These execute **sequentially in program order**, not concurrently. If alloc blocks on any synchronization barrier expecting ST to make progress (slot recycling, m2c queue producer/consumer handoff, etc.), the deadlock is unbreakable from within that wavefront — ST cannot run until alloc returns, alloc cannot return until ST runs.

For `dae_copy_smoke.py` specifically, alloc's first allocate call (`alloc.allocate()`) does succeed (6 free slots, only 2 needed). The block must come later — likely on `m2c.put()` calling `barriers[ptr].arrive_and_wait()` with `numThreadsM2CBarrier = 129`. That barrier expects 1 producer arrival (alloc) + 128 consumer arrivals (compute warps in different wavefronts, 0–127). Compute reaches `[Copy][i=0] before wait` and arrives on c2m, not m2c — so 128 consumer arrivals are still missing from the m2c side.

Why doesn't compute arrive on m2c? Because compute is waiting on c2m for ST to return a freed slot — but ST is structurally blocked behind alloc. Three-way circular wait through three queues, all rooted in the alloc+ST shared wavefront.

## The minimum fix (estimated cost: 1–3 days)

Restructure the kernel layout so each mem-group role gets its own physical wave64 wavefront. Concretely:

### Step 1 — `include/dae/context.cuh`

```cpp
#ifdef __HIP_PLATFORM_AMD__
// AMD wave64: pad each mem warp to a full 64-thread wavefront so alloc, ST,
// LD0, LD1 don't share an EXEC-mask-serialized wavefront.
static constexpr int numThreadsAmdMemWarp = 64;
static constexpr int numThreads =
    numComputeWarps * numThreadsPerWarp +     // 4 * 32 = 128 (2 wave64s)
    numMemoryWarps  * numThreadsAmdMemWarp;   // 4 * 64 = 256 (4 wave64s)
// Total: 384 threads/CTA, well under MI300X's 1024 max.
#else
static constexpr int numThreads = numThreadsPerWarp * (numComputeWarps + numMemoryWarps);
#endif
```

### Step 2 — `include/dae/dae2.cuh`

```cpp
const int sm_id    = blockIdx.x;
const int thread_id = threadIdx.x;

#ifdef __HIP_PLATFORM_AMD__
const int compute_threads = numComputeWarps * numThreadsPerWarp;       // 128
const int mem_thread_idx  = thread_id - compute_threads;               // 0..255 within mem group
const int warp_id  = mem_thread_idx / numThreadsAmdMemWarp;            // 0..3, one per wave64
const int lane_id  = mem_thread_idx % numThreadsAmdMemWarp;            // 0..63 within wavefront
#else
const int warp_id  = (thread_id % 128) / 32;
const int lane_id  = thread_id % 32;
#endif
```

The compute-group dispatch (`if (threadIdx.x < numComputeWarps * 32)`) stays unchanged.

### Step 3 — guard each role's body so only the first 32 lanes do work

`include/dae/pipeline/allocwarp.cuh` and friends are written assuming `lane_id ∈ [0, 31]`. With wave64 we have `lane_id ∈ [0, 63]`, so the extra 32 lanes (32..63) need to no-op. Two ways:

- **(a) inside dae2.cuh's dispatch**: only call the warp's body for lanes 0..31; the other 32 lanes idle.
  ```cpp
  if (warp_id == 0 && lane_id < 32) {
    allocwarp_execute(lane_id, ...);
  }
  ```
  But the body uses `__shfl_sync(0xFFFFFFFF, ...)` with a 32-bit mask. On AMD, `__shfl_sync` ignores the mask and broadcasts wave-wide. With idle lanes 32..63 in the wavefront, `__shfl_sync(mask, val, src)` would read from `src` lane regardless of which lane src is — but if src ≥ 32 was computed as `pc - di.loop_start_pc % 32`, we'd be reading from a valid lane. The math has to be re-audited.

- **(b) inside each warp's body**: make sure every lane-conditioned branch (`if (lane_id == 0)`, `if (lane_id < N)`) implicitly excludes lanes 32..63.

Option (a) is cleaner; option (b) is invasive.

### Step 4 — re-audit every `__shfl_sync` / `__ballot_sync` for wave64 correctness

`include/dae/pipeline/allocwarp.cuh:68` does:
```cpp
uint64_t addr_accum = __shfl_sync(0xFFFFFFFF, di.gpr[1], pc - di.loop_start_pc);
```

The src lane is `pc - di.loop_start_pc`. With the existing 32-lane convention this gives a value in 0..31. On wave64 with idle lanes 32..63 holding undefined `di.gpr[1]`, reading from lane src ∈ 0..31 still works, but any code that does `lane_id >= reg_start && lane_id < reg_end` (line 193) needs to ensure `reg_end ≤ 32` so idle lanes aren't activated.

`__ballot_sync` returns a lane-mask — on wave64 it returns a 64-bit mask, but the codebase treats it as 32-bit. Every consumer of a ballot result needs review.

### Step 5 — re-audit `numThreadsLDBarrier = 2` and friends

Barrier counts assume specific arrival counts. With each mem role on its own wavefront (64 threads), but only one of those threads doing work (`if (lane_id == 0)`), the arrival count stays the same — but it's worth verifying every `barrier_init` call.

## Why this is a real refactor, not a tweak

- `__shfl_sync` semantics differ between wave32 and wave64 — every cross-lane op needs an audit.
- `numThreadsM2CBarrier = numComputeWarps * numThreadsPerWarp + 1` assumes compute = 128 threads. If we ever change `numComputeWarps` for AMD, the barrier count drifts.
- The test for "memory warp group" (`if (threadIdx.x < numComputeWarps * 32)`) hardcodes 32; needs to use `numThreadsPerWarp`.
- `compute_dispatch.cuh` has its own warp/lane assumptions for the actual compute kernels (RMSNorm, SiLU, etc.). Those are out of scope for the smoke test but in scope for any port that uses the megakernel for real work.

## Decision

**Plan A wins.** The staircase finishes the smoke test goal in 2 more rungs (L5 interpreter, L6 Python wiring), each ~3-5 days, each producing a runnable artifact. Plan B is structurally larger and only pays off if the megakernel itself is the destination — but for the immediate goal of "run `dae_copy_smoke.py` on AMD," the staircase route validates each primitive in isolation and avoids dragging the entire megakernel surface along.

Plan B becomes attractive again if:
- We need full feature-parity with upstream (RMSNorm, SiLU, GEMV, attention, RoPE).
- The L5/L6 work surfaces unexpected obstacles.
- A future ROCm/clang ships native async-direct LDS load (`__builtin_amdgcn_global_load_lds` working) — that primitive change might also unblock other megakernel paths.

Until then: continue the staircase.
