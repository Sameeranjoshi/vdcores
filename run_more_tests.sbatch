#!/usr/bin/env bash
#SBATCH --job-name=vdcores-more
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
make HIP=1 HIP_ARCH=gfx942 runtime.o debug=0
HIP=1 HIP_ARCH=gfx942 pip install -e . --no-build-isolation --force-reinstall 2>&1 | tail -5

echo ""
echo "==================== TEST 1: smoke (sanity) ===================="
timeout 30 python -u dae_copy_smoke.py 2>&1 | tail -10 || echo "(smoke FAILED)"

echo ""
echo "==================== TEST 2: tma1d.py (load-only, 16 KB chunks) ===================="
# tma1d uses dae.launch() + dae.bench() — no -l flag needed
timeout 60 python -u app/python/tma1d.py 2>&1 | tail -20 || echo "(tma1d FAILED)"

echo ""
echo "==================== TEST 3: register.py (RegStore/RegLoad no-op) ===================="
timeout 30 python -u app/python/register.py -l 2>&1 | tail -20 || echo "(register FAILED)"

echo ""
echo "=== job ${SLURM_JOB_ID} done ==="
date
