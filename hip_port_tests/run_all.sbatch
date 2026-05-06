#!/usr/bin/env bash
#SBATCH --job-name=vdcores-hip-port
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:12:00
#SBATCH --output=/work1/maryhall/sameeran/work/vdcores/hip_port_tests/logs/vdcores-%j.out
#SBATCH --error=/work1/maryhall/sameeran/work/vdcores/hip_port_tests/logs/vdcores-%j.err

# Single consolidated sbatch for the AMD (hip-port) megakernel tests.
# Builds runtime.o for gfx942, reinstalls the dae python package, then runs
# every AMD-targeted test in order: smoke -> tmacopy -> silu_mul -> mfma-dae
# K+M tiling -> multi-block GEMV -> chunked GEMV (K up to 4096).
#
# Submit: sbatch hip_port_tests/run_all.sbatch
#
# REPO_DIR cascade: explicit override > SLURM submit dir > hardcoded default.

set -eo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-/work1/maryhall/sameeran/work/vdcores}}"
cd "${REPO_DIR}"
mkdir -p hip_port_tests/logs

echo "=== job ${SLURM_JOB_ID} on $(hostname) ==="
date
module purge
module load rocm/7.2.0
source "${REPO_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate

echo "--- build ---"
make HIP=1 HIP_ARCH=gfx942 clean
make HIP=1 HIP_ARCH=gfx942 runtime.o debug=0 2>&1 | tail -10
HIP=1 HIP_ARCH=gfx942 pip install -e . --no-build-isolation --force-reinstall 2>&1 | tail -3

run_test() {
  local label="$1"; shift
  local timeout_s="$1"; shift
  echo ""
  echo "==================== ${label} ===================="
  if timeout "${timeout_s}" "$@" 2>&1 | tail -25; then
    :
  else
    echo "(${label} FAILED, exit=$?)"
  fi
}

# Sanity: 1D memory pipeline through the AMD megakernel interpreter.
run_test "sanity: smoke (single SM, 1D copy)" 30 \
  python -u hip_port_tests/python/dae_copy_smoke.py

# Sanity: 1D load-only pipeline (Dummy compute, no store).
run_test "sanity: tma1d (1D load pipeline)" 30 \
  python -u app/python/tma1d.py -b 10

# Sanity: 1D load+store pipeline (Dummy compute).
run_test "sanity: dae3 (1D load+store pipeline)" 30 \
  python -u app/python/dae3.py -b 10

# Sanity: multi-block memory pipeline (132 SMs, slot reuse, back-pressure).
run_test "sanity: tmacopy (multi-block memory pipeline)" 60 \
  python -u app/python/tmacopy.py -b 10

# Real compute #1: SILU+Mul fused activation (compute wave produces output).
run_test "compute: silu_mul (compute_done signaling)" 60 \
  python -u app/python/silu_mul.py -b 10

# Real compute #2: MFMA bf16 16x16 with K-accumulation + M-tiling.
run_test "compute: mfma-dae K+M tiling" 30 \
  python -u hip_port_tests/python/mfma_dae_test.py

# Real compute #3: multi-block GEMV at K=256 (one atom call per block, scaled to 64 SMs).
run_test "compute: multi-block GEMV (K=256)" 60 \
  python -u hip_port_tests/python/mfma_gemv_test.py

# Real compute #4: chunked GEMV (cross-atom-call K-accumulation, K up to 4096).
# This is the Path A test: K_global = num_chunks * 256, with persistent fp32
# accumulators across chunks and compute_consumed back-pressure on the LD wave.
run_test "compute: chunked GEMV (K up to 4096)" 120 \
  python -u hip_port_tests/python/mfma_gemv4k_test.py

echo ""
echo "=== job ${SLURM_JOB_ID} done ==="
date
