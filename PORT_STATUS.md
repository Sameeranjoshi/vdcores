# VDCores → AMD HIP Port Status

Branch: `hip-port`. Targeting MI300X (gfx942 / CDNA3).

## What's done in this commit

1. **Mechanical hipify pass** — every `.cu` and `.cuh` file under `src/` and `include/` was run through `hipify-perl`. All trivial CUDA runtime APIs are now HIP:
   - `cudaMalloc/Free/Memcpy*` → `hipMalloc/Free/Memcpy*`
   - `cudaStream_t / cudaEvent_t` → `hipStream_t / hipEvent_t`
   - `<cuda_runtime.h>` → `<hip/hip_runtime.h>`
   - `cudaError_t`, `cudaSuccess`, `cudaGetErrorString`, `cudaFuncSetAttribute`, etc.
   - Original CUDA files saved at `build/hipify_log/*.prehip` for diff/rollback.

2. **Makefile dual-target** — `make HIP=1 pyext` selects `hipcc --offload-arch=gfx942`; default still uses `nvcc sm_90a`.

3. **Standalone HIP demo** — `app/hip/vdcores_hip_demo.cpp` shows the VDCores pattern (memory-wave + compute-wave + LDS) compiling with pure HIP. Independent of the runtime; verifies the toolchain works on the AMD node.

## What does NOT compile yet on AMD (and why)

The runtime uses Hopper-only features that hipify cannot translate. These will produce build errors on `hipcc` and need real ports:

| Feature | Files | AMD path |
|---|---|---|
| **TMA (`cuTensorMapEncodeTiled`, `cp_async_bulk`)** | `src/runtime.cu`, `src/torch_runtime.cu`, `include/dae/pipeline/stwarp.cuh`, `include/dae/pipeline/ldwarp.cuh` | No equivalent. Replace with `__builtin_amdgcn_global_load_lds` (CDNA3 LDS DMA) or plain wave loads + `__syncthreads()`. |
| **WGMMA** | `include/task/wgmma.cuh`, `include/task/attention.cuh`, `include/task/gemv.cuh`, `include/dae/compute_dispatch.cuh` | Use **MFMA** intrinsics on CDNA3 (`__builtin_amdgcn_mfma_f32_*`). Different layouts. |
| **`cuda::barrier`, `cuda::ptx`** (libcu++) | `include/dae/context.cuh`, `include/dae/dae2.cuh`, `include/dae/queue.cuh`, `include/dae/virtualcore.cuh`, `include/dae/type.cuh`, `include/task/rope.cuh`, `include/task/attention.cuh` | No HIP equivalent. Replace with manual `__shared__ atomic_int` barriers and `__builtin_amdgcn_*` intrinsics for laneid/clusterid/globaltimer. |
| **Inline PTX (`asm volatile(...)`)** | `include/dae/pipeline/ldwarp.cuh` (15), `include/dae/pipeline/stwarp.cuh` (24), `include/dae/queue.cuh` (11), `include/dae/virtualcore.cuh` (11) | Rewrite with `__builtin_amdgcn_*` intrinsics or AMD inline asm (`s_*`, `v_*`, `ds_*`). |
| **Wavefront size assumption** (32) | scattered, esp. anywhere `blockDim.x` math assumes warp=32 | Adjust block sizes to multiples of 64. |

## To make `make HIP=1 pyext` actually compile

Realistic minimum (per the `vdcores_to_hip.pdf` plan, weeks 3–5):
1. Replace `cuda::barrier` with HIP-portable `__shared__ uint32_t` + atomic spin barriers.
2. Replace `cuda::ptx::get_sreg_*` with `__builtin_amdgcn_*` (laneid via `__lane_id()`, blockid via `blockIdx.x`, globaltimer via `__builtin_readcyclecounter()`).
3. Stub or `#ifdef __HIP_PLATFORM_AMD__` the entire TMA path; teach the launcher to fall back to plain `hipMemcpyAsync`.
4. Stub WGMMA tasks with a clear "not implemented on HIP" runtime error so unrelated tasks (e.g. RMSNorm, SiLU, Argmax) can still compile and run.

## To verify on an AMD node

```bash
git checkout hip-port
module load rocm/6.2.1
cd app/hip && make run         # Standalone demo (works today)
cd ../..  && make HIP=1 pyext  # Full runtime build (will fail with the items above)
```
