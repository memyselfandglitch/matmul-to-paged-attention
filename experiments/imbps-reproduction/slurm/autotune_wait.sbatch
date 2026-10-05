#!/usr/bin/env bash
#SBATCH --job-name=imbps-wait
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --array=0-1%1
#SBATCH --output=autotune-wait-%A_%a.out
#SBATCH --error=autotune-wait-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

readonly WAIT_POLICIES=(ACTIVE PASSIVE)
readonly WAIT_POLICY="${WAIT_POLICIES[${SLURM_ARRAY_TASK_ID}]}"
readonly THREADS=96
readonly CPU_LIST="$(python3 src/select_llc_cpus.py --threads "${THREADS}")"
readonly RESULT_DIR="${PROJECT_ROOT}/results/autotune-wait-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${WAIT_POLICY}-t${THREADS}"

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
  > "${PROJECT_ROOT}/results/cpu-selection-wait-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.json"

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/run_standalone_matrix.py \
    --claim autotune_thread_wait \
    --rounds "${ROUNDS:-2}" \
    --warmups "${WARMUPS:-3}" \
    --iterations "${ITERATIONS:-15}" \
    --threads "${THREADS}" \
    --result-dir "${RESULT_DIR}"

echo "Wait-policy pilot: ${RESULT_DIR}"
