#!/usr/bin/env bash
#SBATCH --job-name=kv-uprof
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=uprof-counters-%j.out
#SBATCH --error=uprof-counters-%j.err

set -euo pipefail

readonly REPO_ROOT="${SLURM_SUBMIT_DIR}"
readonly RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/uprof-${SLURM_JOB_ID}}"
readonly DEFAULT_UPROF_PCM="/data/scratch/deveshisingh/tools/AMDuProf_Linux_x64_5.3.521/bin/AMDuProfPcm"
readonly UPROF_PCM="${UPROF_PCM:-${DEFAULT_UPROF_PCM}}"
cd "${REPO_ROOT}"
mkdir -p "${RESULT_DIR}"

export OMP_NUM_THREADS=1
export BLOCKS="${BLOCKS:-512}"
export HEADS="${HEADS:-32}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export HEAD_DIM="${HEAD_DIM:-128}"
export RUN_LENGTHS="${RUN_LENGTHS:-8,4,3,2,1}"
export BLOCK_SEEDS="${BLOCK_SEEDS:-0,1,2}"
export UPROF_REPEATS="${UPROF_REPEATS:-200}"
export PROFILE_CORE="${PROFILE_CORE:-0}"
export PROFILE_CCX="${PROFILE_CCX:-0}"

if [[ ! -x "${UPROF_PCM}" ]]; then
  echo "AMD uProf PCM not found or not executable: ${UPROF_PCM}" >&2
  exit 1
fi

{
  echo "host=$(hostname)"
  echo "kernel=$(uname -r)"
  echo "job=${SLURM_JOB_ID}"
  echo "profile_core=${PROFILE_CORE}"
  echo "profile_ccx=${PROFILE_CCX}"
  echo "cpus_allowed=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)"
  echo "uprof=$(${UPROF_PCM} --version 2>&1)"
  echo "capabilities=$(getcap "${UPROF_PCM}")"
} > "${RESULT_DIR}/environment.txt"

./scripts/build.sh

# Do not use srun --cpu-bind here: this cluster does not support Slurm CPU
# binding. The runner pins every measured application with taskset instead.
python3 python/run_uprof_counters.py \
  --binary build/paged_kv_study \
  --uprof-pcm "${UPROF_PCM}" \
  --output-dir "${RESULT_DIR}" \
  --blocks "${BLOCKS}" \
  --heads "${HEADS}" \
  --block-size "${BLOCK_SIZE}" \
  --head-dim "${HEAD_DIM}" \
  --run-lengths "${RUN_LENGTHS}" \
  --block-seeds "${BLOCK_SEEDS}" \
  --core "${PROFILE_CORE}" \
  --ccx "${PROFILE_CCX}" \
  --repeats "${UPROF_REPEATS}"

python3 python/analyze_uprof_counters.py \
  "${RESULT_DIR}/uprof-counters.csv" \
  | tee "${RESULT_DIR}/uprof-analysis.txt"

echo "Results: ${RESULT_DIR}"
