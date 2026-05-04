#!/usr/bin/env bash
#SBATCH --job-name=vdcores-mfma-dae
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
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
echo "==================== sanity: smoke ===================="
timeout 30 python -u dae_copy_smoke.py 2>&1 | tail -5 || echo "(smoke FAILED)"

echo ""
echo "==================== sanity: silu_mul ===================="
timeout 60 python -u app/python/silu_mul.py -l 2>&1 | tail -5 || echo "(silu FAILED)"

echo ""
echo "==================== sanity: tmacopy (multi-block memory pipeline) ===================="
timeout 60 python -u app/python/tmacopy.py -l 2>&1 | tail -5 || echo "(tmacopy FAILED)"

echo ""
echo "==================== MFMA in dae2: K-accumulation ===================="
timeout 30 python -u mfma_dae_test.py 2>&1 | tail -20 || echo "(mfma-dae FAILED)"

echo ""
echo "==================== MFMA in dae2: multi-block GEMV ===================="
timeout 60 python -u mfma_gemv_test.py 2>&1 | tail -20 || echo "(mfma-gemv FAILED)"

echo "=== job ${SLURM_JOB_ID} done ==="
date
