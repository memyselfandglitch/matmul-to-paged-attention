#!/usr/bin/env bash
#SBATCH --job-name=imbps-splits
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=autotune-splits-%j.out
#SBATCH --error=autotune-splits-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

readonly THREADS="${AUTOTUNE_THREADS:-96}"
readonly WAIT_POLICY="${AUTOTUNE_WAIT_POLICY:-ACTIVE}"
readonly CPU_LIST="$(python3 src/select_llc_cpus.py --threads "${THREADS}")"
readonly RESULT_PREFIX="${PROJECT_ROOT}/results/autotune-splits-${SLURM_JOB_ID}-${WAIT_POLICY}-t${THREADS}"

export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY="${WAIT_POLICY}"
export LIBXSMM_BLOCK_SIZE=32
export EXPECTED_PHYSICAL_CORES="${SLURM_CPUS_PER_TASK}"
export REQUIRE_SINGLE_SOCKET=1
export REJECT_SMT=1

numactl --cpunodebind=0 --membind=0 ./scripts/preflight.sh
python3 src/select_llc_cpus.py --threads "${THREADS}" --json \
  > "${PROJECT_ROOT}/results/cpu-selection-splits-${SLURM_JOB_ID}.json"

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/run_standalone_matrix.py \
    --claim autotune_split_width \
    --rounds "${ROUNDS:-3}" \
    --warmups "${WARMUPS:-3}" \
    --iterations "${ITERATIONS:-15}" \
    --threads "${THREADS}" \
    --result-dir "${RESULT_PREFIX}-thresholds"

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/run_standalone_matrix.py \
    --claim autotune_equal_rows \
    --rounds "${EQUAL_ROWS_ROUNDS:-3}" \
    --warmups "${WARMUPS:-3}" \
    --iterations "${ITERATIONS:-15}" \
    --threads "${THREADS}" \
    --result-dir "${RESULT_PREFIX}-equal-rows"

echo "Split-width sweep prefix: ${RESULT_PREFIX}"
