#!/usr/bin/env bash
#SBATCH --job-name=vdcores
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --output=vdcores-%j.out
#SBATCH --error=vdcores-%j.err

set -eo pipefail

# --- Locate the repo (directory containing this script) ---
REPO_DIR="/work1/maryhall/sameeran/work/vdcores"
cd "${REPO_DIR}"

echo "=== job ${SLURM_JOB_ID} on $(hostname) ==="
echo "pwd: $(pwd)"
date

# --- Module loads ---
module purge
module load rocm/7.2.0

echo "--- rocm-smi ---"
rocm-smi || true

# --- Conda activation (mirrors the batch.md manual steps) ---
source "${REPO_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate

# ============================================================
# Run commands below
# ============================================================

# Example 1: build & run the HIP demo
make HIP=1 HIP_ARCH=gfx942 clean
make HIP=1 HIP_ARCH=gfx942 runtime.o debug=0
HIP=1 HIP_ARCH=gfx942 pip install -e . --no-build-isolation --force-reinstall
timeout 30 python -u dae_copy_smoke.py 2>&1 | head -300 || echo "(script exited non-zero — likely a hang killed by timeout)"
# Example 2: any python entry point you want, e.g.
# python warmup.py
# python dae_copy_smoke.py

echo "=== job ${SLURM_JOB_ID} done ==="
date
