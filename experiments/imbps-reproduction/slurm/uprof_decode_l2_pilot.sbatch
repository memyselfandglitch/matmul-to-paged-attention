#!/usr/bin/env bash
#SBATCH --job-name=imbps-upr-pilot
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --array=0-1%1
#SBATCH --output=uprof-decode-l2-pilot-%A_%a.out
#SBATCH --error=uprof-decode-l2-pilot-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly PASS_NAMES=(cache traffic)
readonly PASS_NAME="${PASS_NAMES[SLURM_ARRAY_TASK_ID]}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/uprof-decode-l2-pilot-${SLURM_ARRAY_JOB_ID}-${PASS_NAME}"
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
if [[ ! -e /sys/bus/event_source/devices/amd_l3 || \
      ! -e /sys/bus/event_source/devices/amd_df ]]; then
  echo "amd_l3 and amd_df perf devices are required" >&2
  exit 1
fi

readonly THREADS="${SLURM_CPUS_PER_TASK}"
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
    --require-thp always \
    --expected-l2-mib 96 \
    --output "${RESULT_ROOT}/runtime-validation.json"
numactl --physcpubind="${CPU_LIST}" --membind=0 ./scripts/preflight.sh

if [[ "${PASS_NAME}" == cache ]]; then
  readonly METRICS="ipc,l2,l3"
else
  readonly METRICS="memory"
fi

readonly PROBE_CSV="${RESULT_ROOT}/option-probe.csv"
readonly PROBE_LOG="${RESULT_ROOT}/option-probe.log"
set +e
"${UPROF_BIN}" -X -m "${METRICS}" -c package=0 -C \
  -o "${PROBE_CSV}" -- /bin/sleep 2 > "${PROBE_LOG}" 2>&1
readonly PROBE_STATUS=$?
set -e
if (( PROBE_STATUS != 0 )); then
  cat "${PROBE_LOG}" >&2
  echo "uProf option probe failed for ${METRICS}" >&2
  exit "${PROBE_STATUS}"
fi

python3 src/run_uprof_matrix.py \
  --claim decode_l2_pilot_opt30b \
  --pass-name "${PASS_NAME}" \
  --uprof-bin "${UPROF_BIN}" \
  --package 0 \
  --cpu-list "${CPU_LIST}" \
  --threads "${THREADS}" \
  --rounds 3 \
  --warmups 3 \
  --iterations 5 \
  --minimum-counter-seconds 3 \
  --seed 20261006 \
  --result-dir "${RESULT_ROOT}/matrix"

echo "Decode L2 pilot uProf ${PASS_NAME}: ${RESULT_ROOT}/matrix"
