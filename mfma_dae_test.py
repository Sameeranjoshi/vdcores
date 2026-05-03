#!/usr/bin/env python3
# MFMA bf16 16x16 matmul through the dae2 interpreter on AMD.
#
# Wires the validated MFMA path (app/hip/mfma_matmul.cpp) into the dae2
# interpreter as a hardcoded compute mode, exercised end-to-end from the
# Python launcher: LD wave loads A and B, compute wave runs one MFMA call,
# ST wave stores the bf16 result. Verified against torch.matmul.
#
# This is the bridge between the standalone HIP MFMA validation and a real
# upstream GEMV/GEMM port.

import sys; sys.argv = ['x', '-l']
import torch
from dae.launcher import (
    Launcher, ComputeInstruction, TmaLoad1D, TmaStore1D, TerminateC, TerminateM,
)
from dae.util import dae_app
from dae.runtime import opcode as op_module

# The AMD interpreter hijacks OP_GEMM_M64N64 as the marker for our MFMA
# bf16 16x16 matmul demo (kept inside the supported-compute-ops set so the
# launcher validation passes, while no upstream python test uses it).
OP_AMD_DEBUG_MATMUL_BF16_16x16 = int(op_module.OP_GEMM_M64N64)
gpu = torch.device('cuda')

torch.manual_seed(0xC0FFEE)
A = (torch.rand(16, 16, dtype=torch.bfloat16, device=gpu) - 0.5)
B = (torch.rand(16, 16, dtype=torch.bfloat16, device=gpu) - 0.5)
C = torch.zeros(16, 16, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms=1, device=gpu)


def task(sm: int):
    return [
        ComputeInstruction(opcode=OP_AMD_DEBUG_MATMUL_BF16_16x16, args=[]),
        TmaStore1D(C, bytes=16 * 16 * 2),
        TmaLoad1D(A, bytes=16 * 16 * 2),
        TmaLoad1D(B, bytes=16 * 16 * 2),
    ]


dae.i(task, TerminateC(), TerminateM())
dae_app(dae)
torch.cuda.synchronize()

ref = (A.to(torch.float32) @ B.to(torch.float32)).to(torch.bfloat16)
diff = (C.to(torch.float32) - ref.to(torch.float32)).abs()
max_abs = diff.max().item()
mean_abs = diff.mean().item()

print(f"C[0, :4]   = {C[0, :4].tolist()}")
print(f"ref[0, :4] = {ref[0, :4].tolist()}")
print(f"max_abs_err = {max_abs:.6f}")
print(f"mean_abs_err = {mean_abs:.6f}")
ok = max_abs < 0.05  # bf16 noise tolerance
print(f"mfma-dae: {'PASS' if ok else 'FAIL'}")
