# hip-port AMD megakernel tests

All AMD/MI300X (gfx942) test artifacts that exercise the megakernel through
the `dae` Python frontend live here. Standalone HIP demos (the L0/L1/L2/L3
staircase, mfma_smoke, mfma_matmul) live under `app/hip/` instead — they are
self-contained C++/HIP binaries and have their own `Makefile` + `run_staircase.sbatch`.

## Layout

```
hip_port_tests/
├── README.md             # this file
├── run_all.sbatch        # one consolidated SLURM job that runs every test below
├── logs/                 # SLURM output drops here (created on first run)
└── python/
    ├── dae_copy_smoke.py # single-SM 1D copy through the AMD interpreter
    ├── mfma_dae_test.py  # MFMA bf16 16x16 with K-accumulation + M-tiling
    ├── mfma_gemv_test.py # multi-block GEMV at K=256 (one atom per block)
    └── mfma_gemv4k_test.py  # chunked GEMV (cross-atom-call K-accumulation, K up to 4096)
```

## Running

From the repo root:

```sh
sbatch hip_port_tests/run_all.sbatch
```

The sbatch picks up `REPO_DIR` from (in order): explicit env override, `SLURM_SUBMIT_DIR`,
or a hardcoded default. It builds `runtime.o` for gfx942, reinstalls the `dae` package,
then runs every test in order with per-test timeouts. Output lands in
`hip_port_tests/logs/vdcores-<jobid>.out`.

Individual tests can also be run directly once the package is installed:

```sh
python -u hip_port_tests/python/mfma_gemv4k_test.py
```

The `app/python/silu_mul.py` and `app/python/tmacopy.py` invocations in the
sbatch are upstream NVIDIA Python entry points — on AMD they go through our
hip-port interpreter unchanged, which is exactly the cross-platform validation
we want.

## What each test exercises

| Test                  | Path                            | What it validates                                                       |
|-----------------------|---------------------------------|-------------------------------------------------------------------------|
| smoke                 | LD → ST                         | minimal MInst loop, 1D global→LDS→global copy                          |
| tmacopy               | LD → ST × 132 SMs               | multi-block memory pipeline, slot reuse, back-pressure                  |
| silu_mul              | LD A,B → compute → ST           | compute wave producer/consumer, `compute_done` signaling, fp32 ops      |
| mfma-dae K+M tiling   | LD A,B → MFMA → ST              | bf16 16×16×16 MFMA, K-accumulation chain, M-tile outer loop             |
| multi-block GEMV K=256| × 64 SMs                        | full Gemv_M64N8 atom shape per block, scaling across SMs                |
| chunked GEMV K≤4096   | streaming chunks per block      | Path A: cross-atom-call accumulation, `compute_consumed` back-pressure  |

All tests are bit-exact against `torch.matmul` (or torch reference for SILU)
on a GPU-only timing path — `dae.bench()` reads the on-device profile counters
written by the AMD megakernel, no Python launcher overhead in the numbers.
