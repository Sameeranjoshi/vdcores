#!/usr/bin/env bash
#SBATCH --job-name=vdcores-mfma
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:03:00
#SBATCH --output=vdcores-%j.out
#SBATCH --error=vdcores-%j.err

set -eo pipefail

REPO_DIR="/work1/maryhall/sameeran/work/vdcores"
cd "${REPO_DIR}/app/hip"

echo "=== job ${SLURM_JOB_ID} on $(hostname) ==="
date
module purge
module load rocm/7.2.0

echo "--- build mfma_smoke ---"
make ARCH=gfx942 mfma_smoke 2>&1 | tail -20

echo ""
echo "--- run mfma_smoke ---"
timeout 10 ./mfma_smoke || echo "(mfma_smoke exited non-zero)"

echo "=== job ${SLURM_JOB_ID} done ==="
date
