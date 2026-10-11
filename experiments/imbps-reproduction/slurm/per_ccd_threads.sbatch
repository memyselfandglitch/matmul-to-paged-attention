#!/usr/bin/env bash
#SBATCH --job-name=imbps-ccd-thread
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --array=0-10%1
#SBATCH --output=per-ccd-threads-%A_%a.out
#SBATCH --error=per-ccd-threads-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly SOCKET_THREADS="${SLURM_CPUS_PER_TASK}"
readonly INNER_THREADS="${PER_CCD_INNER_THREADS:-8}"
readonly NUMA_NODE="${PER_CCD_NUMA_NODE:-0}"
readonly NODE_CPU_LIST_PATH="/sys/devices/system/node/node${NUMA_NODE}/cpulist"
readonly CLAIM="${PER_CCD_CLAIM:-opt30b_pthread_parallel_imbps}"
readonly VARIANTS=(
  tpp-all96
  global-k12-all96
  global-k24-all96
  global-k48-all96
  thread-p1-k1
  thread-p2-k2
  thread-p4-k4
  thread-p6-k6
  thread-p12-k12
  thread-p12-k24
  thread-p12-k48
)
readonly VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/per-ccd-threads-${SLURM_ARRAY_JOB_ID}"
readonly CASE_DIR="${RESULT_ROOT}/${VARIANT}"

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

BENCH_THREADS="${SOCKET_THREADS}"
if [[ "${VARIANT}" == thread-* ]]; then
  BENCH_THREADS="${INNER_THREADS}"
  export OMP_PROC_BIND=false
  unset OMP_PLACES || true
else
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
fi
readonly BENCH_THREADS
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

optional_args=()
if [[ -n "${PER_CCD_ROWS:-}" ]]; then
  optional_args+=(--rows "${PER_CCD_ROWS}")
fi

numactl --physcpubind="${CPU_LIST}" --membind="${NUMA_NODE}" \
  python3 src/per_ccd_mlp_bench.py \
    --claim "${CLAIM}" \
    --variant "${VARIANT}" \
    --rounds "${PER_CCD_ROUNDS:-7}" \
    --warmups "${PER_CCD_WARMUPS:-3}" \
    --threads "${BENCH_THREADS}" \
    "${optional_args[@]}" \
    --output "${CASE_DIR}/result.json"

echo "Threaded per-CCD result: ${CASE_DIR}/result.json"
