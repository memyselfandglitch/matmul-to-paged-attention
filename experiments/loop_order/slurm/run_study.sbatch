#!/usr/bin/env bash
#SBATCH --job-name=matmul-study
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=loop-order-%j.out
#SBATCH --error=loop-order-%j.err

set -euo pipefail

readonly STUDY_ROOT="${STUDY_ROOT:-${SLURM_SUBMIT_DIR}}"
readonly RESULT_DIR="${STUDY_ROOT}/results/job-${SLURM_JOB_ID}"
readonly MAX_N="${MAX_N:-1920}"
readonly REPETITIONS="${REPETITIONS:-5}"
readonly MIN_N="${MIN_N:-384}"

cd "${STUDY_ROOT}"
mkdir -p "${RESULT_DIR}"

{
  echo "host=$(hostname)"
  echo "job=${SLURM_JOB_ID}"
  echo "cpu=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
  c++ --version | sed -n '1p'
  echo "max_n=${MAX_N}"
  echo "min_n=${MIN_N}"
  echo "repetitions=${REPETITIONS}"
} >"${RESULT_DIR}/environment.txt"

make all

./build/loop_order_study "${MAX_N}" "${REPETITIONS}" "${MIN_N}" \
  | tee "${RESULT_DIR}/loop-order.txt"

./build/matmul_optimization_stages "${MAX_N}" "${REPETITIONS}" "${MIN_N}" \
  | tee "${RESULT_DIR}/optimization-stages.txt"
