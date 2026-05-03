#!/usr/bin/env bash
#SBATCH --job-name=vdcores-silu
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:08:00
#SBATCH --output=vdcores-%j.out
#SBATCH --error=vdcores-%j.err

set -eo pipefail

REPO_DIR="/work1/maryhall/sameeran/work/vdcores"
cd "${REPO_DIR}"

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

echo ""
echo "==================== TEST 1: smoke (sanity) ===================="
timeout 30 python -u dae_copy_smoke.py 2>&1 | tail -10 || echo "(smoke FAILED)"

echo ""
echo "==================== TEST 2: tmacopy (sanity) ===================="
timeout 60 python -u app/python/tmacopy.py -l 2>&1 | tail -10 || echo "(tmacopy FAILED)"

echo ""
echo "==================== TEST 3: silu_mul.py (REAL COMPUTE) ===================="
timeout 60 python -u app/python/silu_mul.py -l 2>&1 | tail -30 || echo "(silu FAILED)"

echo ""
echo "=== job ${SLURM_JOB_ID} done ==="
date
