#!/usr/bin/env python3
# MFMA bf16 (M_total x N=16) = (M_total x K_total) @ (K_total x 16) matmul
# through the dae2 interpreter on AMD.
#
# Tests:
#   K-accumulation: chain K_iters MFMAs, feed fp32 accumulator forward.
#   M-tiling:       outer loop over M_blocks 16-row tiles.
#
# K = K_iters * 16, M_total = M_blocks * 16. N is fixed at 16.
# Verified against torch.matmul.

import sys
import torch
from dae.launcher import (
    Launcher, ComputeInstruction, TmaLoad1D, TmaStore1D, TerminateC, TerminateM,
)
from dae.runtime import opcode as op_module

OP_AMD_DEBUG_MATMUL_BF16 = int(op_module.OP_GEMM_M64N64)
gpu = torch.device('cuda')


def run_one(M_blocks: int, K_iters: int, seed: int = 0):
    torch.manual_seed(seed)
    M = M_blocks * 16
    K = K_iters * 16
    A = (torch.rand(M, K,  dtype=torch.bfloat16, device=gpu) - 0.5)
    B = (torch.rand(K, 16, dtype=torch.bfloat16, device=gpu) - 0.5)
    C = torch.zeros(M, 16, dtype=torch.bfloat16, device=gpu)

    dae = Launcher(num_sms=1, device=gpu)

    def task(sm: int):
        return [
            ComputeInstruction(opcode=OP_AMD_DEBUG_MATMUL_BF16,
                               args=[K_iters, M_blocks]),
            TmaStore1D(C, bytes=M * 16 * 2),
            TmaLoad1D(A, bytes=M * K * 2),
            TmaLoad1D(B, bytes=K * 16 * 2),
        ]

    dae.i(task, TerminateC(), TerminateM())
    dae.launch()
    torch.cuda.synchronize()

    ref = (A.to(torch.float32) @ B.to(torch.float32)).to(torch.bfloat16)
    diff = (C.to(torch.float32) - ref.to(torch.float32)).abs()
    return diff.max().item(), diff.mean().item(), C, ref


print("MFMA in dae2 — K-accumulation + M-tiling tests")
print("=" * 70)

# (M_blocks, K_iters, label, tolerance)
cases = [
    (1,  1, "K=16  M=16  (baseline single MFMA)",                0.02),
    (1,  4, "K=64  M=16  (4 K-MFMAs accumulating)",              0.10),
    (1, 16, "K=256 M=16  (16 K-MFMAs, matches Gemv_M64N8 K)",    0.50),
    (2, 16, "K=256 M=32  (M-tiling × 2 + K=256)",                0.50),
    (4,  8, "K=128 M=64  (M-tiling × 4 + K=128, fills slot A)",  0.50),
]

results = []
for M_blocks, K_iters, label, tol in cases:
    max_e, mean_e, C, ref = run_one(M_blocks, K_iters, seed=K_iters * 100 + M_blocks)
    ok = max_e < tol
    status = "PASS" if ok else "FAIL"
    print(f"  {label:55s}  max={max_e:.6f}  mean={mean_e:.6f}  {status}")
    if not ok:
        print(f"    C  [0,:4] = {C[0, :4].tolist()}")
        print(f"    ref[0,:4] = {ref[0, :4].tolist()}")
    results.append(ok)

all_ok = all(results)
print("=" * 70)
print(f"mfma-dae K+M tiling: {'ALL PASS' if all_ok else 'SOME FAILED'}")
sys.exit(0 if all_ok else 1)
