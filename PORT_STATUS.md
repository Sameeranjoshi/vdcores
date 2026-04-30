# VDCores → AMD HIP Port Status

Branch: `hip-port`. Targeting MI300X (gfx942 / CDNA3).

## Status snapshot

| Stage | NVIDIA Hopper | AMD MI300X (this branch) |
|---|---|---|
| `make pyext` (CUDA) | ✅ Works | n/a |
| `make HIP=1 runtime.o` | n/a | ✅ **Compiles** (as of commit on this branch) |
| `make HIP=1 pyext` | n/a | ⚠️  Compiles runtime; PyTorch bindings (`src/torch_runtime.cu`) still need TMA-bytes replacement |
| Standalone HIP demo (`app/hip/`) | n/a | ✅ Compiles + runs on MI300X |
| Functional kernels on AMD | ✅ All | ❌ All compute kernels are **stubs that trap** — see below |

## What's actually done

1. **Mechanical hipify pass** — every `.cu` / `.cuh` under `src/` and `include/` translated:
   - `cuda*` → `hip*` runtime APIs, `<cuda_runtime.h>` → `<hip/hip_runtime.h>`.
   - Originals archived at `build/hipify_log/*.prehip`.

2. **Dual-target Makefile** — `make HIP=1 ...` selects `hipcc --offload-arch=gfx942` plus `-D__AMDGCN_WAVEFRONT_SIZE=64 -fPIC`; CUDA path unchanged.

3. **HIP compatibility shim** at `include/dae/hip_compat.cuh`:
   - `cuda::barrier<thread_scope_block>` — phase-counted block barrier built on a single `unsigned long long` shared atomic. Functional for two-warp queue handoff.
   - `cuda::ptx::get_sreg_*` — mapped to `__builtin_amdgcn_*` (laneid, clusterid via `blockIdx.x`, globaltimer via `__builtin_readcyclecounter()`).
   - `cuda::ptx::cp_async_bulk*`, `fence_proxy_async` — TMA paths stubbed (`__builtin_trap`); fence degrades to `__threadfence_block`.
   - `__shfl_sync` / `__ballot_sync` / `__any_sync` etc. → HIP non-sync forms.
   - `cuda::aligned_size_t`, `cuda::device::memcpy_async_tx`, `barrier_expect_tx` — TMA helpers stubbed.
   - `__hip_bfloat16` / `__hip_bfloat162` aliases (CUDA naming → HIP naming).
   - `cutlass::bfloat16_t`, `cute::SM90_*<...>`, `cute::GMMA::Major`, `Int<>`, `make_shape`, `tile_to_shape` — typedef/struct stubs purely for name lookup so dispatch headers parse.
   - `__nanosleep` → `__builtin_amdgcn_s_sleep`.
   - `make_bfloat162`, `__cvta_generic_to_shared` shims.

4. **Standalone HIP demo** — `app/hip/vdcores_hip_demo.cpp` implements the memory-wave + compute-wave + LDS pattern in pure HIP. Compiles + runs.

5. **Stubbed kernels under `__HIP_PLATFORM_AMD__`** — every Hopper-specific kernel now has a variadic-template stub that calls `__builtin_trap()` if exercised. Keeps the runtime infrastructure (scheduler, queues, allocator, dispatch) compiling cleanly while making "silently produces garbage" impossible:
   - `task_attention_fwd_flash3_grouped(_mma)`, `task_split_post_reduce` (attention.cuh)
   - `task_gemm` (wgmma.cuh)
   - `task_gemv`, `task_gemv_mma` (gemv.cuh)
   - `task_silu_smem`, `task_silu_smem_1D` (silu.cuh)
   - `task_rope_interleaved` (rope.cuh)
   - `task_rms_norm_f16_from_smem` (rms_norm.cuh) — relied on bf16x2 intrinsics
   - `ldwarp_execute_singlethread` (TMA load pipeline, ldwarp.cuh)
   - `stwarp_execute_singlethread` (TMA store pipeline, stwarp.cuh)
   - `create_tma_descriptor` (host, runtime.cu) — returns zero-init descriptor

## What's left to make it functional (not just compile)

Replace each stub above with a real AMD implementation:

| Stub | What to write |
|---|---|
| `ldwarp_execute_singlethread` | `__builtin_amdgcn_global_load_lds` plus a `__shared__` atomic counter for the "tx complete" handshake. CDNA3 has the LDS DMA, just not as a high-level descriptor. |
| `stwarp_execute_singlethread` | Plain wave-cooperative store loops (`ds_read` / global write). No bulk-store equivalent. |
| `task_gemm` (WGMMA) | MFMA intrinsics (`__builtin_amdgcn_mfma_f32_*`). Layout differs from WGMMA — A/B fragments are 16-wide on CDNA3. |
| `task_gemv*` | Same MFMA idea, narrow N. |
| `task_attention_fwd_flash3_*` | Either rebuild on MFMA or call rocBLAS / Composable Kernel. The flash3 control flow (split-K, post-reduce) can be reused. |
| `task_silu_smem*`, `task_rope_interleaved` | Hand-roll without CuTe layouts. These don't need MMA, they're just elementwise + reduce. |
| `task_rms_norm_f16_from_smem` | Same — write a bf16 path using fp32 reductions and `hip_bfloat16` element conversion. No bf16x2 vector unit on MI300X exposed via intrinsics. |
| `create_tma_descriptor` | No-op or remove entirely. Replace TMA opcodes in `compute_dispatch.cuh` with `hipMemcpyAsync` + LDS staging on the AMD path. |
| `cuda::barrier` shim | Validate correctness under contention (currently the `s_sleep(1)` poll is fine; sanity-test the phase rollover). |

## Estimated effort

- **rms_norm + silu + rope** (no MMA): a couple of days each. These unblock pre-/post-attention pipeline.
- **rocBLAS-based gemm + attention** (skip MFMA hand-rolling): ~1 week — gives correct results but loses the VDCores DAE pipeline overlap.
- **MFMA-based gemm + flash-attention rewrite**: 3–4 weeks for parity with the Hopper version.
- **TMA → LDS-DMA pipeline**: 1–2 weeks; the trickiest because the whole VDCores model assumes async TMA.

## Build & verify

```bash
# Setup (only needed once on a node without /opt/rocm-6.2.1):
source /scratch/general/vast/u1418973/vdcores/miniconda3/etc/profile.d/conda.sh
conda create -n hip -c conda-forge -y hipcc rocm-device-libs    # ~5 min
conda activate hip
unset CXXFLAGS CFLAGS CPPFLAGS LDFLAGS

# Standalone demo:
cd /scratch/general/vast/u1418973/vdcores/app/hip && make
./vdcores_hip_demo            # needs an AMD GPU; on Hopper-only nodes this will report "no ROCm-capable device"

# Full runtime object (compiles anywhere with hipcc):
cd /scratch/general/vast/u1418973/vdcores
make HIP=1 runtime.o          # produces a 26KB ELF for gfx942

# pyext (PyTorch bindings) — torch_runtime.cu still needs TMA-byte input handling
# stubbed before this will succeed end-to-end.
make HIP=1 pyext              # not yet
```
