#!/usr/bin/env python3
# GEMV_WGMMA family test: Gemv_M64N8 with cross-atom K-accumulation.
#
# Uses Gemv_M64N8(kTiles=num_chunks) to run cross-chunk K-accumulation
# with persistent fp32 accumulators. Tests the AMD interpreter's handling
# of OP_GEMV_WGMMA__M_64__N_8__K_256__BLOAD_4 and variants.
#
# Same layout and structure as mfma_gemv4k_test.py but uses the
# Gemv_M64N8 instruction (with family_ref opcode generation).

import sys
import numpy as np
import torch
from dae.launcher import (
    Launcher, Gemv_M64N8, TmaLoad1D, TmaStore1D, TerminateC, TerminateM,
)

gpu = torch.device('cuda')

# Per-block atom shape (matches Gemv_M64N8 atom on the AMD side).
M_PER_BLOCK = 64
K_PER_CHUNK = 256
N           = 16

NS_PER_TICK = 10.0   # MI300X wall_clock64 = 100 MHz


def run(num_blocks: int, num_chunks: int, n_iters: int = 50, seed: int = 0):
    torch.manual_seed(seed)
    K = K_PER_CHUNK * num_chunks   # K_global per block
    M_total = num_blocks * M_PER_BLOCK

    # Native chunked layouts so each chunk's TmaLoad1D source is contiguous.
    # A[c, m, p] holds the p-th K-elem of chunk c for row m: slicing
    #   A[c, m_lo:m_hi, :] is contiguous in (M, K) row-major.
    # B[c, n, p] holds the p-th K-elem of chunk c for column n: slicing
    #   B[c, :, :] is contiguous in (N, K) row-major. This matches the
    #   layout AMD MFMA expects (sB[n * K_chunk + k]) and what 2D TMA
    #   loads of an app/python (N, K) matB produce after the K-block fold.
    A_chunked = (torch.rand(num_chunks, M_total, K_PER_CHUNK,
                            dtype=torch.bfloat16, device=gpu) - 0.5)
    B_chunked = (torch.rand(num_chunks, N, K_PER_CHUNK,
                            dtype=torch.bfloat16, device=gpu) - 0.5)
    C = torch.zeros(M_total, N, dtype=torch.bfloat16, device=gpu)

    # Reference: assemble the logical (M, K) and (K, N) views.
    A_logical = A_chunked.permute(1, 0, 2).contiguous().reshape(M_total, K)
    # B_chunked is (num_chunks, N, K_PER_CHUNK). Permute to (num_chunks,
    # K_PER_CHUNK, N) so chunk-major K → outer index, then flatten chunks
    # into the K dimension to get (K, N).
    B_logical = B_chunked.permute(0, 2, 1).reshape(K, N).contiguous()

    dae = Launcher(num_sms=num_blocks, device=gpu)

    def task(sm: int):
        m_lo = sm * M_PER_BLOCK
        m_hi = m_lo + M_PER_BLOCK
        insts = [
            Gemv_M64N8(kTiles=num_chunks),
            TmaStore1D(C[m_lo:m_hi, :], bytes=M_PER_BLOCK * N * 2),
        ]
        for c in range(num_chunks):
            insts.append(TmaLoad1D(A_chunked[c, m_lo:m_hi, :],
                                   bytes=M_PER_BLOCK * K_PER_CHUNK * 2))
            insts.append(TmaLoad1D(B_chunked[c, :, :],
                                   bytes=K_PER_CHUNK * N * 2))
        return insts

    dae.i(task, TerminateC(), TerminateM())

    # Warm-up + correctness check.
    dae.launch()
    torch.cuda.synchronize()

    ref = (A_logical.to(torch.float32) @ B_logical.to(torch.float32)).to(torch.bfloat16)
    diff = (C.to(torch.float32) - ref.to(torch.float32)).abs()
    max_e, mean_e = diff.max().item(), diff.mean().item()

    # Timed iterations using on-device profile counters (matches dae.bench()).
    execution_times_ns = []
    for _ in range(n_iters):
        dae.launch()
        prof_u64 = dae.profile[:num_blocks, 0:2].cpu().numpy()
        delta_ticks = int(prof_u64[:, 1].max()) - int(prof_u64[:, 0].min())
        execution_times_ns.append(delta_ticks * NS_PER_TICK)
    execution_times_ns = np.asarray(execution_times_ns, dtype=np.float64)
    avg_ns    = float(execution_times_ns.mean())
    min_ns    = float(execution_times_ns.min())
    median_ns = float(np.median(execution_times_ns))
    avg_ms    = avg_ns / 1e6

    flops_per_iter = 2.0 * M_total * K * N
    tflops = flops_per_iter / (avg_ns * 1e-9) / 1e12
    bytes_per_iter = 2 * (M_total * K + K * N + M_total * N)
    bw_gbs = bytes_per_iter / (avg_ns * 1e-9) / 1e9

    return {
        "M_total": M_total, "K": K, "N": N, "blocks": num_blocks,
        "num_chunks": num_chunks,
        "max_err": max_e, "mean_err": mean_e,
        "avg_ms": avg_ms, "min_ns": min_ns, "median_ns": median_ns,
        "tflops": tflops, "bw_gbs": bw_gbs,
    }


print("GEMV_WGMMA: Gemv_M64N8 with cross-chunk K-accumulation (K=num_chunks*256, N=16) on MI300X")
print("=" * 100)
results = []
# Sweep: 1, 2, 4, 8, 16 chunks (K=256 .. K=4096) on a fixed block count.
for num_chunks in [1, 2, 4, 8, 16]:
    r = run(num_blocks=64, num_chunks=num_chunks, seed=num_chunks)
    # Larger K → larger bf16 noise. Tolerance scales with sqrt(K).
    tol = 0.5 * (r["K"] / 256.0) ** 0.5 + 0.5
    ok = r["max_err"] < tol
    status = "PASS" if ok else "FAIL"
    print(f"  chunks={r['num_chunks']:2d}  K={r['K']:5d}  M={r['M_total']:5d}  "
          f"max_err={r['max_err']:.4f} (tol={tol:.2f})  "
          f"min={r['min_ns']/1e3:8.2f}us  med={r['median_ns']/1e3:8.2f}us  "
          f"avg={r['avg_ms']:.4f}ms  "
          f"throughput={r['tflops']:.2f} TFLOPS  bw={r['bw_gbs']:.1f} GB/s  {status}")
    results.append(ok)

all_ok = all(results)
print("=" * 100)
print(f"GEMV_WGMMA: {'ALL PASS' if all_ok else 'SOME FAILED'}")
sys.exit(0 if all_ok else 1)
