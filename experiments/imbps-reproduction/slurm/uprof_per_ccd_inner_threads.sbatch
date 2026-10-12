#!/usr/bin/env bash
#SBATCH --job-name=imbps-ccd-inner-l3
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --array=0-2%1
#SBATCH --output=uprof-per-ccd-inner-%A_%a.out
#SBATCH --error=uprof-per-ccd-inner-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly SOCKET_THREADS="${SLURM_CPUS_PER_TASK}"
readonly NUMA_NODE="${PER_CCD_NUMA_NODE:-0}"
readonly NODE_CPU_LIST_PATH="/sys/devices/system/node/node${NUMA_NODE}/cpulist"
readonly CLAIM="opt30b_per_ccd_inner_threads"
readonly VARIANTS=(
  tpp-all96
  global-k6-all96
  thread-p12-k12-sharded-t8
)
readonly INNER_THREAD_COUNTS=(96 96 8)
readonly VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
readonly BENCH_THREADS="${INNER_THREAD_COUNTS[${SLURM_ARRAY_TASK_ID}]}"
readonly ACTIVE_ROWS=4096
readonly RESULT_ROOT="${PROJECT_ROOT}/results/uprof-per-ccd-inner-threads-${SLURM_ARRAY_JOB_ID}"
readonly CASE_DIR="${RESULT_ROOT}/${VARIANT}-m${ACTIVE_ROWS}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
source ./scripts/enable_tcmalloc.sh

if [[ -n "${AMDUPROFPCM_BIN:-}" && -x "${AMDUPROFPCM_BIN}" ]]; then
  readonly UPROF_BIN="${AMDUPROFPCM_BIN}"
else
  readonly UPROF_BIN="$(
    find "/data/scratch/${USER}/tools" -type f -name AMDuProfPcm -perm -u+x \
      2>/dev/null | sort -V | tail -1
  )"
fi
if [[ -z "${UPROF_BIN}" || ! -x "${UPROF_BIN}" ]]; then
  echo "AMDuProfPcm was not found" >&2
  exit 1
fi
if [[ ! -d /sys/bus/event_source/devices/amd_l3 ]]; then
  echo "amd_l3 perf device is required" >&2
  exit 1
fi
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

mkdir -p "${RESULT_ROOT}"
numactl --physcpubind="${CPU_LIST}" --membind="${NUMA_NODE}" \
  python3 src/validate_runtime.py \
    --require-tcmalloc \
    --require-thp "${REQUIRE_THP_MODE:-always}" \
    --expected-l2-mib 96 \
    --expected-l3-mib 384 \
    --output "${RESULT_ROOT}/runtime-${VARIANT}-m${ACTIVE_ROWS}.json"

numactl --physcpubind="${CPU_LIST}" --membind="${NUMA_NODE}" \
  python3 src/run_uprof_per_ccd_case.py \
    --uprof-bin "${UPROF_BIN}" \
    --metrics ipc,l2,l3 \
    --package 0 \
    --aggregation ccd \
    --cpu-list "${CPU_LIST}" \
    --claim "${CLAIM}" \
    --variant "${VARIANT}" \
    --rows "${ACTIVE_ROWS}" \
    --threads "${BENCH_THREADS}" \
    --warmups "${PER_CCD_UPROF_WARMUPS:-20}" \
    --rounds "${PER_CCD_UPROF_ROUNDS:-30}" \
    --min-measurement-seconds "${PER_CCD_UPROF_MIN_SECONDS:-5}" \
    --result-dir "${CASE_DIR}"

echo "Per-CCD inner-thread profile: ${CASE_DIR}"
