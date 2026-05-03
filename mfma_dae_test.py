#!/usr/bin/env python3
# MFMA bf16 16x16xK matmul through the dae2 interpreter on AMD.
#
# Tests the K-accumulation loop: each kernel runs K_iters back-to-back
# v_mfma_f32_16x16x16bf16_1k calls, feeding the fp32 accumulator from one
# call into the next. K = K_iters * 16. Verified against torch.matmul.
#
# K=16  -> 1 MFMA (single-shot, same as the original baseline)
# K=64  -> 4 MFMAs
# K=256 -> 16 MFMAs (matches the K dim of the upstream Gemv_M64N8 atom)

import sys
import torch
from dae.launcher import (
    Launcher, ComputeInstruction, TmaLoad1D, TmaStore1D, TerminateC, TerminateM,
)
from dae.runtime import opcode as op_module

OP_AMD_DEBUG_MATMUL_BF16 = int(op_module.OP_GEMM_M64N64)
gpu = torch.device('cuda')


def run_one(K_iters: int, seed: int = 0):
    torch.manual_seed(seed)
    K = K_iters * 16
    A = (torch.rand(16, K,  dtype=torch.bfloat16, device=gpu) - 0.5)
    B = (torch.rand(K,  16, dtype=torch.bfloat16, device=gpu) - 0.5)
    C = torch.zeros(16, 16, dtype=torch.bfloat16, device=gpu)

    dae = Launcher(num_sms=1, device=gpu)

    def task(sm: int):
        return [
            ComputeInstruction(opcode=OP_AMD_DEBUG_MATMUL_BF16, args=[K_iters]),
            TmaStore1D(C, bytes=16 * 16 * 2),
            TmaLoad1D(A, bytes=16 * K * 2),
            TmaLoad1D(B, bytes=K * 16 * 2),
        ]

    dae.i(task, TerminateC(), TerminateM())
    dae.launch()
    torch.cuda.synchronize()

    ref = (A.to(torch.float32) @ B.to(torch.float32)).to(torch.bfloat16)
    diff = (C.to(torch.float32) - ref.to(torch.float32)).abs()
    return diff.max().item(), diff.mean().item(), C, ref


print("MFMA in dae2 — K-accumulation tests")
print("=" * 60)
results = []
for K_iters in [1, 4, 16]:
    max_e, mean_e, C, ref = run_one(K_iters, seed=K_iters)
    K = K_iters * 16
    # bf16 final-cast tolerance grows roughly as sqrt(K) of the per-mac noise.
    # Empirically K=256 mismatches stay well under 0.5 with random ~0.5 inputs.
    tol = 0.5 if K_iters >= 16 else (0.1 if K_iters >= 4 else 0.02)
    ok = max_e < tol
    status = "PASS" if ok else "FAIL"
    print(f"K={K:4d}  max_err={max_e:.6f}  mean_err={mean_e:.6f}  tol={tol:.2f}  {status}")
    if not ok:
        print(f"  C  [0,:4] = {C[0, :4].tolist()}")
        print(f"  ref[0,:4] = {ref[0, :4].tolist()}")
    results.append(ok)

all_ok = all(results)
print("=" * 60)
print(f"mfma-dae K-accumulation: {'ALL PASS' if all_ok else 'SOME FAILED'}")
sys.exit(0 if all_ok else 1)
