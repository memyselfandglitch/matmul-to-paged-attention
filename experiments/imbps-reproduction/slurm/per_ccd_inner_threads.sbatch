#!/usr/bin/env bash
#SBATCH --job-name=imbps-ccd-inner
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --array=0-13%1
#SBATCH --output=per-ccd-inner-%A_%a.out
#SBATCH --error=per-ccd-inner-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly SOCKET_THREADS="${SLURM_CPUS_PER_TASK}"
readonly NUMA_NODE="${PER_CCD_NUMA_NODE:-0}"
readonly NODE_CPU_LIST_PATH="/sys/devices/system/node/node${NUMA_NODE}/cpulist"
readonly CLAIM="opt30b_per_ccd_inner_threads"
readonly VARIANTS=(
  tpp-all96
  global-k6-all96
  thread-p12-k12-sharded-t1
  thread-p12-k12-sharded-t2
  thread-p12-k12-sharded-t4
  thread-p12-k12-sharded-t6
  thread-p12-k12-sharded-t8
)
readonly INNER_THREAD_COUNTS=(96 96 1 2 4 6 8)
readonly ROWS=(2048 4096)
readonly VARIANT_COUNT="${#VARIANTS[@]}"
readonly VARIANT_INDEX="$((SLURM_ARRAY_TASK_ID % VARIANT_COUNT))"
readonly ROW_INDEX="$((SLURM_ARRAY_TASK_ID / VARIANT_COUNT))"
readonly VARIANT="${VARIANTS[${VARIANT_INDEX}]}"
readonly BENCH_THREADS="${INNER_THREAD_COUNTS[${VARIANT_INDEX}]}"
readonly ACTIVE_ROWS="${ROWS[${ROW_INDEX}]}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/per-ccd-inner-threads-${SLURM_ARRAY_JOB_ID}"
readonly CASE_DIR="${RESULT_ROOT}/${VARIANT}-m${ACTIVE_ROWS}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
source ./scripts/enable_tcmalloc.sh

if [[ ! -r "${NODE_CPU_LIST_PATH}" ]]; then
  echo "NUMA-node CPU list is unavailable: ${NODE_CPU_LIST_PATH}" >&2
  exit 1
fi
NODE_CPU_LIST="$(<"${NODE_CPU_LIST_PATH}")"
readonly NODE_CPU_LIST
CPU_LIST="$(taskset --cpu-list "${NODE_CPU_LIST}" \
  python3 src/select_llc_cpus.py \
    --threads "${SOCKET_THREADS}" --expected-groups 12 --expected-group-size 8)"
readonly CPU_LIST

if [[ "${VARIANT}" == thread-* ]]; then
  export OMP_PROC_BIND=false
  unset OMP_PLACES || true
else
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
fi
export OMP_NUM_THREADS="${BENCH_THREADS}"
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=ACTIVE
export NUMEXPR_MAX_THREADS="${BENCH_THREADS}"
export NUMEXPR_NUM_THREADS="${BENCH_THREADS}"
export LIBXSMM_BLOCK_SIZE=32

mkdir -p "${CASE_DIR}"
numactl --physcpubind="${CPU_LIST}" --membind="${NUMA_NODE}" \
  python3 src/validate_runtime.py \
    --require-tcmalloc \
    --require-thp "${REQUIRE_THP_MODE:-always}" \
    --expected-l2-mib 96 \
    --expected-l3-mib 384 \
    --output "${CASE_DIR}/runtime-validation.json"

taskset --cpu-list "${NODE_CPU_LIST}" \
  python3 src/select_llc_cpus.py \
    --threads "${SOCKET_THREADS}" \
    --expected-groups 12 \
    --expected-group-size 8 \
    --json > "${CASE_DIR}/cpu-selection.json"

numactl --physcpubind="${CPU_LIST}" --membind="${NUMA_NODE}" \
  python3 src/per_ccd_mlp_bench.py \
    --claim "${CLAIM}" \
    --variant "${VARIANT}" \
    --rows "${ACTIVE_ROWS}" \
    --rounds "${PER_CCD_INNER_ROUNDS:-30}" \
    --warmups "${PER_CCD_INNER_WARMUPS:-20}" \
    --min-measurement-seconds "${PER_CCD_INNER_MIN_SECONDS:-5}" \
    --threads "${BENCH_THREADS}" \
    --output "${CASE_DIR}/result.json"

echo "Per-CCD inner-thread result: ${CASE_DIR}/result.json"
