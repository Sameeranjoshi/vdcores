#!/usr/bin/env python3
# 2D TMA smoke test: load a (N, K) bf16 tile via wgmma_load (rank-3 K-major
# descriptor) into LDS slot A, store the LDS contents back to a 1D output
# buffer with TmaStore1D, and verify the bytes match the source matrix.
#
# This isolates the new AmdTmaDesc / multi-D box-copy path without involving
# any compute kernel.
#
# K-major rank-3 descriptor decomposes K into (K_block_count, blockK):
#   - blockK = 128 / elsize (= 64 for bf16)
#   - global_dims = [blockK, N, K // blockK]
#   - K-blocks are stride-contiguous in memory (gstride[1] == blockK*elsize)
#
# The AMD multi-D box walk in dae2.cuh detects this contiguity and folds
# dims 0 + 2 into a single contiguous matrix-row copy per N-row, producing
# row-major (N, K) LDS layout that matches matrix-natural memory order.
# So out = src.flatten() exactly.

import sys
import torch
from dae.launcher import (
    Launcher, TmaTensor, TmaStore1D, TerminateC, TerminateM,
)
from dae.tma_utils import Major

gpu = torch.device('cuda')

N, K = 64, 256
EL = 2  # bf16
BLOCK_K = 128 // EL  # 64

src = (torch.arange(N * K, dtype=torch.float32, device=gpu) / 1024.0).to(torch.bfloat16)
src = src.reshape(N, K).contiguous()
out = torch.zeros(N * K, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms=1, device=gpu)

# wgmma_load(N, K, Major.K) -> rank-3 descriptor with box (blockK, N, K/blockK).
load = TmaTensor(dae, src).wgmma_load(N, K, Major.K)
print(f"opcode=0x{load.opcode:x}  rank={load.rank}  size={load.size} bytes")

def task(sm: int):
    return [
        load.cord(0, 0),                      # box origin (n=0, k=0)
        TmaStore1D(out, bytes=N * K * EL),    # dump LDS slot A linearly
    ]

dae.i(task, TerminateC(), TerminateM())
dae.launch()
torch.cuda.synchronize()

# Expected LDS layout: row-major (N, K) — matches src.flatten() because the
# AMD box walk folds the contiguous K-block dim into a per-row contiguous copy.
expected = src.flatten()

diff = (expected.to(torch.float32) - out.to(torch.float32)).abs()
max_e = diff.max().item()
mean_e = diff.mean().item()

print(f"src.flatten()[:8]  = {src.flatten()[:8].tolist()}")
print(f"expected[:8]       = {expected[:8].tolist()}")
print(f"out[:8]            = {out[:8].tolist()}")
print(f"max_err  = {max_e:.6f}")
print(f"mean_err = {mean_e:.6f}")

ok = max_e < 1e-3
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
