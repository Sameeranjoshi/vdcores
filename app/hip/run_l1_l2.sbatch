#!/usr/bin/env bash
#SBATCH --job-name=vdcores-hip
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:03:00
#SBATCH --output=vdcores-hip-%j.out
#SBATCH --error=vdcores-hip-%j.err

set -eo pipefail
REPO_DIR="/work1/maryhall/sameeran/work/vdcores"
cd "${REPO_DIR}/app/hip"

echo "=== job ${SLURM_JOB_ID} on $(hostname) ==="
date
module purge
module load rocm/7.2.0
echo "--- rocm-smi ---"
rocm-smi || true

echo "--- make clean & build (only sources that exist will compile) ---"
make clean
# Build whatever sources are present; skip missing ones gracefully.
for tgt in vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2; do
  if [ -f "${tgt}.cpp" ]; then
    echo "--- build ${tgt} ---"
    make "${tgt}"
  else
    echo "--- skip ${tgt} (no source) ---"
  fi
done

for bin in vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2; do
  if [ -x "./${bin}" ]; then
    echo "--- run ${bin} ---"
    timeout 10 "./${bin}" || echo "(${bin} exited non-zero)"
  fi
done

echo "=== job ${SLURM_JOB_ID} done ==="
date
