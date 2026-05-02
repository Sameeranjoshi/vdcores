#!/usr/bin/env bash
#SBATCH --job-name=vdcores-tmacopy
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
echo "pwd: $(pwd)"
date

module purge
module load rocm/7.2.0

echo "--- rocm-smi ---"
rocm-smi || true

source "${REPO_DIR}/miniconda3/etc/profile.d/conda.sh"
conda activate

echo "--- build runtime ---"
make HIP=1 HIP_ARCH=gfx942 clean
make HIP=1 HIP_ARCH=gfx942 runtime.o debug=0
HIP=1 HIP_ARCH=gfx942 pip install -e . --no-build-isolation --force-reinstall

echo "--- smoke (sanity baseline) ---"
timeout 30 python -u dae_copy_smoke.py 2>&1 | tail -20 || echo "(smoke failed)"

echo "--- tmacopy (132 SMs, 1024 loads, 16 KB each) ---"
timeout 120 python -u app/python/tmacopy.py -l 2>&1 | tail -40 || echo "(tmacopy exited non-zero — likely hang killed by timeout)"

echo "=== job ${SLURM_JOB_ID} done ==="
date
