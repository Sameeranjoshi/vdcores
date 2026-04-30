# VDCores HIP demo (MI300X)

Minimal proof-of-concept showing the VDCores decoupled memory/compute model on AMD HIP.
One CTA = one **memory wavefront** + one **compute wavefront**, communicating through LDS.

Workload: `c = a + b` over chunks staged into LDS.

## Files
- `vdcores_hip_demo.cpp` — kernel + host driver (HIP)
- `Makefile` — builds with `hipcc --offload-arch=gfx942` (MI300X / CDNA3)

## Build & run on MI300X
```bash
module load rocm        # or whatever your env uses
make
make run
```
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
| `sm_90a`                      | `gfx942` via `--offload-arch`      |

## What's intentionally NOT here (per scope)

- No async global→LDS intrinsic (`__builtin_amdgcn_global_load_lds`) — plain LDS staging
- No double-buffering / pipelining
- No GEMM / attention / RoPE — just vector add
- No Python / PyTorch wrapper — pure C++
- No multi-block scheduling or queue infrastructure

This is a *one-kernel* sanity check that the VDCores partitioning idea (split a CTA
into memory-only and compute-only wavefronts, sync via LDS) compiles and runs on
ROCm/HIP. Extend from here.
