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
for candidate in "${PYTHON_BIN:-}" "${HOME}/envs/vllm-rocm721/bin/python" python3; do
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
  echo "Set PYTHON_BIN to a Python executable that can import torch." >&2
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
