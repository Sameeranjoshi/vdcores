#!/usr/bin/env bash
set -eo pipefail
# Note: -u (nounset) is intentionally OFF — conda's activate.d hooks
# (e.g. ~cuda-nvcc_activate.sh) reference unset env vars like NVCC_PREPEND_FLAGS.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${REPO_DIR}/miniconda3"
INSTALLER="${REPO_DIR}/miniconda.sh"

if [ ! -d "${CONDA_DIR}" ]; then
    wget -O "${INSTALLER}" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh"
    bash "${INSTALLER}" -b -p "${CONDA_DIR}"
    rm -f "${INSTALLER}"
fi

# Activate for this shell only — do NOT touch ~/.bashrc
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda install -y -c conda-forge python cuda-toolkit=13.0.2 cutlass
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy transformers accelerate

echo
echo "Done. To use this env in a new shell:"
echo "  source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate"
