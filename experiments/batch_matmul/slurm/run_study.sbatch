#!/usr/bin/env bash
#SBATCH --job-name=bmm-study
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=batch-matmul-%j.out
#SBATCH --error=batch-matmul-%j.err

set -euo pipefail

readonly STUDY_ROOT="${STUDY_ROOT:-${SLURM_SUBMIT_DIR}}"
readonly RESULT_DIR="${STUDY_ROOT}/results/job-${SLURM_JOB_ID}"
readonly REPETITIONS="${REPETITIONS:-21}"
readonly SWEEP_REPETITIONS="${SWEEP_REPETITIONS:-5}"

cd "${STUDY_ROOT}"
mkdir -p "${RESULT_DIR}"

python_exec=""
readonly USER_SCRATCH="/data/scratch/${USER}"
python_candidates=(
  "${PYTHON_BIN:-}"
  "${USER_SCRATCH}/envs/vllm-rocm721/bin/python"
  "${USER_SCRATCH}/pace-venv/bin/python"
  "${USER_SCRATCH}/miniforge3/bin/python"
  "${HOME}/envs/vllm-rocm721/bin/python"
  "${HOME}/pace-venv/bin/python"
  "${HOME}/miniforge3/bin/python"
  python3
)

# Include any other conda/venv interpreters created under the user's scratch
# directory. nullglob prevents an unmatched pattern from becoming a fake path.
shopt -s nullglob
for candidate in "${USER_SCRATCH}"/envs/*/bin/python "${HOME}"/envs/*/bin/python; do
  python_candidates+=("${candidate}")
done
shopt -u nullglob

for candidate in "${python_candidates[@]}"; do
  if [[ -z "${candidate}" ]]; then
    continue
  fi
  if "${candidate}" -c 'import torch' >/dev/null 2>&1; then
    python_exec="${candidate}"
    break
  fi
done

if [[ -z "${python_exec}" ]]; then
  echo "error: no Python interpreter with PyTorch was found" >&2
  echo "Tried:" >&2
  printf '  %s\n' "${python_candidates[@]}" >&2
  echo "Set PYTHON_BIN to a Python executable that can run: import torch" >&2
  exit 1
fi
readonly python_exec

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

{
  echo "host=$(hostname)"
  echo "job=${SLURM_JOB_ID}"
  echo "cpu=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
  echo "python=${python_exec}"
  "${python_exec}" --version
  "${python_exec}" -c 'import torch; print(f"torch={torch.__version__}")'
  echo "cpu_threads=${SLURM_CPUS_PER_TASK:-4}"
  echo "repetitions=${REPETITIONS}"
  echo "sweep_repetitions=${SWEEP_REPETITIONS}"
} >"${RESULT_DIR}/environment.txt"

"${python_exec}" bmm_kv_study.py \
  --device cpu \
  --repetitions "${REPETITIONS}" \
  --cache-sweep \
  --sweep-repetitions "${SWEEP_REPETITIONS}" \
  --csv "${RESULT_DIR}/cache-sweep.csv" \
  | tee "${RESULT_DIR}/bmm-study.txt"
