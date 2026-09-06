#!/bin/bash
#SBATCH --job-name=vllm-rocm-setup
#SBATCH --partition=jobmn01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/rhome/deveshisingh/vllm-rocm-setup-%j.out
#SBATCH --error=/rhome/deveshisingh/vllm-rocm-setup-%j.err

set -euo pipefail

readonly USER_ROOT="/rhome/deveshisingh"
readonly SOURCE_ROOT="/data/scratch/deveshisingh/matmul-to-paged-attention"
readonly SHARED_ROOT="${USER_ROOT}/matmul-to-paged-attention"
readonly MINIFORGE_ROOT="${USER_ROOT}/miniforge3"
readonly ENV_ROOT="${USER_ROOT}/envs/vllm-rocm721"
readonly INSTALLER="/data/scratch/deveshisingh/Miniforge3.sh"

echo "Node: $(hostname)"
if [[ ! -f "${SHARED_ROOT}/experiments/vllm/pyproject.toml" ]]; then
  echo "Copying the repository to shared storage: ${SHARED_ROOT}"
  mkdir -p "${SHARED_ROOT}"
  rsync -a \
    --exclude='.venv' \
    --exclude='build' \
    --exclude='results' \
    "${SOURCE_ROOT}/" "${SHARED_ROOT}/"
else
  echo "Using the existing shared repository: ${SHARED_ROOT}"
fi

if [[ ! -x "${MINIFORGE_ROOT}/bin/conda" ]]; then
  echo "Installing Miniforge under shared storage"
  bash "${INSTALLER}" -b -p "${MINIFORGE_ROOT}"
fi

source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"

if [[ -x "${ENV_ROOT}/bin/python" ]] && \
   ! "${ENV_ROOT}/bin/python" -c 'import torch; assert torch.version.hip' \
     >/dev/null 2>&1; then
  echo "Removing the existing non-ROCm environment"
  conda remove --yes --all --prefix "${ENV_ROOT}"
fi

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  echo "Creating the Python 3.12 environment"
  conda create --yes --prefix "${ENV_ROOT}" python=3.12 pip
fi

"${ENV_ROOT}/bin/python" -m pip install --upgrade pip uv
"${ENV_ROOT}/bin/uv" pip install \
  --python "${ENV_ROOT}/bin/python" \
  --upgrade \
  --prerelease=allow \
  vllm \
  --extra-index-url https://wheels.vllm.ai/rocm/nightly/rocm721

mkdir -p "${SHARED_ROOT}/experiments/vllm_layout/results"

echo "Installation verification"
"${ENV_ROOT}/bin/python" - <<'PY'
import importlib.metadata
import sys

print("Python:", sys.version.replace("\n", " "))
print("PyTorch package:", importlib.metadata.version("torch"))
print("vLLM package:", importlib.metadata.version("vllm"))
print("Runtime import verification is performed on the GPU node.")
PY

echo "Environment: ${ENV_ROOT}"
echo "Shared repository: ${SHARED_ROOT}"
