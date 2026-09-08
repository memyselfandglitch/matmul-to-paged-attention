#!/usr/bin/env bash
#SBATCH --job-name=kv-crossover
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=crossover-%j.out
#SBATCH --error=crossover-%j.err

set -euo pipefail

readonly REPO_ROOT="${SLURM_SUBMIT_DIR}"
readonly RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/crossover-${SLURM_JOB_ID}}"
cd "${REPO_ROOT}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
export BLOCKS="${BLOCKS:-512}"
export HEADS="${HEADS:-32}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export HEAD_DIM="${HEAD_DIM:-128}"
export RUN_LENGTHS="${RUN_LENGTHS:-512,256,128,64,32,16,8,4,2,1}"
export TRIALS="${TRIALS:-5}"
export WARMUPS="${WARMUPS:-2}"
export REPEATS="${REPEATS:-9}"

./scripts/build.sh

srun --cpu-bind=cores python3 python/run_crossover.py \
  --binary build/paged_kv_study \
  --output "${RESULT_DIR}/crossover-raw.csv" \
  --blocks "${BLOCKS}" \
  --heads "${HEADS}" \
  --block-size "${BLOCK_SIZE}" \
  --head-dim "${HEAD_DIM}" \
  --run-lengths "${RUN_LENGTHS}" \
  --trials "${TRIALS}" \
  --warmups "${WARMUPS}" \
  --repeats "${REPEATS}"

python3 python/analyze_crossover.py \
  "${RESULT_DIR}/crossover-raw.csv" \
  --summary-csv "${RESULT_DIR}/crossover-summary.csv" \
  | tee "${RESULT_DIR}/crossover-analysis.txt"

echo "Results: ${RESULT_DIR}"
