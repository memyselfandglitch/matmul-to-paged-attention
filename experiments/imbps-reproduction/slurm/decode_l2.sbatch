#!/usr/bin/env bash
#SBATCH --job-name=imbps-decode-l2
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=decode-l2-%j.out
#SBATCH --error=decode-l2-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly THREADS="${SLURM_CPUS_PER_TASK}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/decode-l2-${SLURM_JOB_ID}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate
source ./scripts/enable_tcmalloc.sh

readonly CPU_LIST="$(python3 src/select_llc_cpus.py --threads "${THREADS}")"
export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=ACTIVE
export NUMEXPR_MAX_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export LIBXSMM_BLOCK_SIZE=32
export EXPECTED_PHYSICAL_CORES="${THREADS}"
export REQUIRE_SINGLE_SOCKET=1
export REJECT_SMT=1

mkdir -p "${RESULT_ROOT}"
numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/validate_runtime.py \
    --require-tcmalloc \
    --require-thp "${REQUIRE_THP_MODE:-always}" \
    --expected-l2-mib 96 \
    --output "${RESULT_ROOT}/runtime-validation.json"
numactl --physcpubind="${CPU_LIST}" --membind=0 ./scripts/preflight.sh
python3 src/select_llc_cpus.py --threads "${THREADS}" --json \
  > "${RESULT_ROOT}/cpu-selection.json"

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/run_standalone_matrix.py \
    --claim decode_l2_opt30b \
    --rounds "${ROUNDS:-5}" \
    --warmups "${WARMUPS:-5}" \
    --iterations "${ITERATIONS:-20}" \
    --min-measurement-seconds "${MIN_MEASUREMENT_SECONDS:-1}" \
    --threads "${THREADS}" \
    --result-dir "${RESULT_ROOT}/timing"

echo "Decode L2 timing: ${RESULT_ROOT}/timing"
