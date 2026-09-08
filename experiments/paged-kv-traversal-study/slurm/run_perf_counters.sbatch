#!/usr/bin/env bash
#SBATCH --job-name=kv-perf
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=perf-counters-%j.out
#SBATCH --error=perf-counters-%j.err

set -euo pipefail

readonly REPO_ROOT="${SLURM_SUBMIT_DIR}"
readonly RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/perf-${SLURM_JOB_ID}}"
cd "${REPO_ROOT}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
export BLOCKS="${BLOCKS:-512}"
export HEADS="${HEADS:-32}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export HEAD_DIM="${HEAD_DIM:-128}"
export RUN_LENGTHS="${RUN_LENGTHS:-512,32,16,8,4,1}"
export BLOCK_SEED="${BLOCK_SEED:-0}"
export PERF_REPEATS="${PERF_REPEATS:-100}"

./scripts/build.sh

srun --cpu-bind=cores python3 python/run_perf_counters.py \
  --binary build/paged_kv_study \
  --output "${RESULT_DIR}/perf-counters.csv" \
  --blocks "${BLOCKS}" \
  --heads "${HEADS}" \
  --block-size "${BLOCK_SIZE}" \
  --head-dim "${HEAD_DIM}" \
  --run-lengths "${RUN_LENGTHS}" \
  --block-seed "${BLOCK_SEED}" \
  --repeats "${PERF_REPEATS}"

python3 python/analyze_perf_counters.py \
  "${RESULT_DIR}/perf-counters.csv" \
  | tee "${RESULT_DIR}/perf-analysis.txt"

echo "Results: ${RESULT_DIR}"
