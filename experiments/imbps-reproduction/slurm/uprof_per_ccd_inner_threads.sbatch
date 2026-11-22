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
readonly CLAIM="${PER_CCD_UPROF_CLAIM:-opt30b_per_ccd_inner_threads}"
readonly LABEL="${PER_CCD_UPROF_RESULT_LABEL:-uprof-per-ccd-inner-threads}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/${LABEL}-${SLURM_ARRAY_JOB_ID}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
source ./scripts/enable_tcmalloc.sh

batch_args=()
if [[ "${CLAIM}" == opt30b_per_ccd_batch_layout ]]; then
  shape="$(python3 src/summarize_ccd_batch_layout.py \
    --shape-index "${SLURM_ARRAY_TASK_ID}" --shapes "${PER_CCD_LAYOUT_SHAPES:?}")"
  read -r batch rows_per_item <<< "${shape}"
  active_rows="$((batch * rows_per_item))"
  case_name="b${batch}-q${rows_per_item}"
  variants=(batch-items-t8 thread-p12-k12-sharded-t8 tpp-all96 global-k6-all96)
  batch_args=(--batch-items "${batch}")
  repeats="${PER_CCD_UPROF_REPEATS:-4}"
  warmups="${PER_CCD_UPROF_WARMUPS:-3}"
  iterations="${PER_CCD_UPROF_ROUNDS:-10}"
else
  variants=(tpp-all96 global-k6-all96 thread-p12-k12-sharded-t8)
  variants=("${variants[SLURM_ARRAY_TASK_ID]}")
  active_rows=4096
  case_name="${variants[0]}-m${active_rows}"
  repeats=1
  warmups="${PER_CCD_UPROF_WARMUPS:-20}"
  iterations="${PER_CCD_UPROF_ROUNDS:-30}"
fi
collect_traffic=false
if [[ "${CLAIM}" == opt30b_per_ccd_batch_layout ]] && \
   (( active_rows == 3072 || active_rows == 6144 )); then
  collect_traffic=true
fi

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
if "${collect_traffic}" && [[ ! -d /sys/bus/event_source/devices/amd_df ]]; then
  echo "amd_df perf device is required for the DRAM traffic pass" >&2
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

export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_WAIT_POLICY=ACTIVE
export LIBXSMM_BLOCK_SIZE=32

mkdir -p "${RESULT_ROOT}"
numactl --physcpubind="${CPU_LIST}" --membind="${NUMA_NODE}" \
  python3 src/validate_runtime.py \
    --require-tcmalloc \
    --require-thp "${REQUIRE_THP_MODE:-always}" \
    --expected-l2-mib 96 \
    --expected-l3-mib 384 \
    --output "${RESULT_ROOT}/runtime-${case_name}.json"

profile_case() (
  local variant="$1" result_dir="$2" seed="$3" metrics="$4" aggregation="$5"
  local threads="${SOCKET_THREADS}"
  if [[ "${variant}" == thread-* || "${variant}" == batch-items-* ]]; then
    threads=8
    export OMP_PROC_BIND=false
    unset OMP_PLACES
  else
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
  fi
  export OMP_NUM_THREADS="${threads}"
  export NUMEXPR_MAX_THREADS="${threads}"
  export NUMEXPR_NUM_THREADS="${threads}"
  numactl --physcpubind="${CPU_LIST}" --membind="${NUMA_NODE}" \
    python3 src/run_uprof_per_ccd_case.py \
      --uprof-bin "${UPROF_BIN}" --metrics "${metrics}" \
      --package 0 --aggregation "${aggregation}" --cpu-list "${CPU_LIST}" \
      --claim "${CLAIM}" --variant "${variant}" --rows "${active_rows}" \
      --threads "${threads}" --warmups "${warmups}" --rounds "${iterations}" \
      --seed "${seed}" --min-measurement-seconds "${PER_CCD_UPROF_MIN_SECONDS:-5}" \
      --result-dir "${result_dir}" "${batch_args[@]}"
)

for ((round=1; round<=repeats; round++)); do
  for ((offset=0; offset<${#variants[@]}; offset++)); do
    variant="${variants[(round + SLURM_ARRAY_TASK_ID + offset) % ${#variants[@]}]}"
    profile_name="${case_name}"
    if [[ "${CLAIM}" == opt30b_per_ccd_batch_layout ]]; then
      profile_name="${case_name}-r${round}-${variant}"
    fi
    seed="$((20251000 + round))"
    profile_case "${variant}" "${RESULT_ROOT}/${profile_name}" "${seed}" ipc,l2,l3 ccd
    if "${collect_traffic}"; then
      profile_case "${variant}" "${RESULT_ROOT}/traffic/${profile_name}" "${seed}" memory none
    fi
  done
done
