#!/usr/bin/env python3
# Multi-block GEMV-shape matmul through the dae2 interpreter on AMD.
#
# Each block runs the full M=64, K=256 atom shape on its own row-tile of A,
# loading the shared B matrix and writing its 64-row slice of C. With
# num_blocks SMs, total M = num_blocks * 64 rows.
#
# Workload  : C = A @ B  where  A is (M_total × K)  bf16
#                                B is (K × 16)      bf16
#                                C is (M_total × 16) bf16
# Per block : M=64 K=256 N=16 (one Gemv_M64N8-shaped atom call)
# Verified vs torch.matmul. Reports kernel time for AMD/NVIDIA comparison.

import sys
import numpy as np
import torch
from dae.launcher import (
    Launcher, ComputeInstruction, TmaLoad1D, TmaStore1D, TerminateC, TerminateM,
)
from dae.runtime import opcode as op_module

OP_AMD_DEBUG_MATMUL_BF16 = int(op_module.OP_GEMM_M64N64)
gpu = torch.device('cuda')

# AMD wall-clock counter rate. On MI300X this is 100 MHz → 10 ns/tick.
# hipDeviceAttributeWallClockRate returns kHz; ns_per_tick = 1e6 / kHz.
def _wallclock_ns_per_tick():
    try:
        # PyTorch's HIP path exposes this attr through torch.cuda.
        props = torch.cuda.get_device_properties(0)
        # Fallback: most ROCm builds don't expose wall_clock_rate via props,
        # so just hardcode the documented 100 MHz for gfx942.
        return 10.0
    except Exception:
        return 10.0

NS_PER_TICK = _wallclock_ns_per_tick()

# Full Gemv_M64N8 atom shape per block.
M_PER_BLOCK = 64
K           = 256
N           = 16
K_ITERS     = K // 16    # = 16 K-MFMAs per block
M_BLOCKS    = M_PER_BLOCK // 16   # = 4 row-tiles per block (matches our compute)


def run(num_blocks: int, n_iters: int = 50, seed: int = 0):
    torch.manual_seed(seed)
    M_total = num_blocks * M_PER_BLOCK
    A = (torch.rand(M_total, K, dtype=torch.bfloat16, device=gpu) - 0.5)
    # B is laid out (N, K) row-major in memory. AMD MFMA reads sB[n*K+k].
    B = (torch.rand(N,       K, dtype=torch.bfloat16, device=gpu) - 0.5)
    C = torch.zeros(M_total, N, dtype=torch.bfloat16, device=gpu)

    dae = Launcher(num_sms=num_blocks, device=gpu)

    def task(sm: int):
        m_lo = sm * M_PER_BLOCK
        m_hi = m_lo + M_PER_BLOCK
        return [
            ComputeInstruction(opcode=OP_AMD_DEBUG_MATMUL_BF16,
                               args=[K_ITERS, M_BLOCKS]),
            TmaStore1D(C[m_lo:m_hi, :], bytes=M_PER_BLOCK * N * 2),
            TmaLoad1D(A[m_lo:m_hi, :], bytes=M_PER_BLOCK * K * 2),
            TmaLoad1D(B,                bytes=K * N * 2),
        ]

    dae.i(task, TerminateC(), TerminateM())

    # Warm-up + correctness check (1 launch).
    dae.launch()
    torch.cuda.synchronize()

    # C = A @ B^T because B is now (N, K) instead of (K, N).
    ref = (A.to(torch.float32) @ B.to(torch.float32).T).to(torch.bfloat16)
    diff = (C.to(torch.float32) - ref.to(torch.float32)).abs()
    max_e, mean_e = diff.max().item(), diff.mean().item()

    # Timed iterations using on-device profile counters (dae.profile slots
    # 0,1 = start,end nanoseconds per SM, written by the AMD megakernel).
    # This excludes Python launcher overhead, matching the inbuilt
    # dae.bench() path used by the upstream NVIDIA apps.
    execution_times_ns = []
    for _ in range(n_iters):
        dae.launch()
        # Profile slots are AMD wall-clock ticks (s_memrealtime, 100 MHz).
        # Take max(end) - min(start) across SMs to capture full kernel span.
        prof_u64 = dae.profile[:num_blocks, 0:2].cpu().numpy()
        delta_ticks = int(prof_u64[:, 1].max()) - int(prof_u64[:, 0].min())
        execution_times_ns.append(delta_ticks * NS_PER_TICK)
    execution_times_ns = np.asarray(execution_times_ns, dtype=np.float64)
    avg_ns = float(execution_times_ns.mean())
    min_ns = float(execution_times_ns.min())
    median_ns = float(np.median(execution_times_ns))
    avg_ms = avg_ns / 1e6

    # Throughput: per-iteration work = M_total * K * N MACs * 2 (FMA = 2 flops).
    flops_per_iter = 2.0 * M_total * K * N
    tflops = flops_per_iter / (avg_ns * 1e-9) / 1e12

    # Per-iter memory traffic: A + B (loaded) + C (stored) bf16 bytes.
    bytes_per_iter = 2 * (M_total * K + K * N + M_total * N)
    bw_gbs = bytes_per_iter / (avg_ns * 1e-9) / 1e9

    return {
        "M_total": M_total, "K": K, "N": N, "blocks": num_blocks,
        "max_err": max_e, "mean_err": mean_e,
        "avg_ms": avg_ms, "min_ns": min_ns, "median_ns": median_ns,
        "tflops": tflops, "bw_gbs": bw_gbs,
    }


print("Multi-block GEMV (M=64 K=256 N=16 per block) on MI300X")
print("=" * 80)
results = []
for num_blocks in [1, 8, 32, 64]:
    r = run(num_blocks, seed=num_blocks)
    ok = r["max_err"] < 1.0  # bf16 noise tolerance for K=256
    status = "PASS" if ok else "FAIL"
    print(f"  blocks={r['blocks']:3d}  M_total={r['M_total']:5d}  "
          f"max_err={r['max_err']:.4f}  mean_err={r['mean_err']:.6f}  "
          f"min={r['min_ns']/1e3:7.2f}us  med={r['median_ns']/1e3:7.2f}us  "
          f"avg={r['avg_ms']:.4f}ms  "
          f"throughput={r['tflops']:.2f} TFLOPS  "
          f"bw={r['bw_gbs']:.1f} GB/s  {status}")
    results.append(ok)

all_ok = all(results)
print("=" * 80)
print(f"multi-block GEMV: {'ALL PASS' if all_ok else 'SOME FAILED'}")
sys.exit(0 if all_ok else 1)
