#!/usr/bin/env bash
#SBATCH --job-name=imbps-l2-core
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=uprof-decode-l2-core-%j.out
#SBATCH --error=uprof-decode-l2-core-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/uprof-decode-l2-core-${SLURM_JOB_ID}"
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
    --require-thp "${REQUIRE_THP_MODE:-always}" \
    --expected-l2-mib 96 \
    --output "${RESULT_ROOT}/runtime-validation.json"
numactl --physcpubind="${CPU_LIST}" --membind=0 ./scripts/preflight.sh

readonly METRICS="ipc,l2"
readonly PROBE_CSV="${RESULT_ROOT}/option-probe.csv"
readonly PROBE_LOG="${RESULT_ROOT}/option-probe.log"
set +e
"${UPROF_BIN}" -X -m "${METRICS}" -c package=0 -C \
  -o "${PROBE_CSV}" -- /bin/sleep 2 > "${PROBE_LOG}" 2>&1
readonly PROBE_STATUS=$?
set -e
if (( PROBE_STATUS != 0 )); then
  cat "${PROBE_LOG}" >&2
  echo "uProf core-only option probe failed for ${METRICS}" >&2
  exit "${PROBE_STATUS}"
fi

python3 src/run_uprof_matrix.py \
  --claim decode_l2_core_pilot_opt30b \
  --pass-name l2 \
  --uprof-bin "${UPROF_BIN}" \
  --package 0 \
  --cpu-list "${CPU_LIST}" \
  --threads "${THREADS}" \
  --rounds "${ROUNDS:-3}" \
  --warmups "${WARMUPS:-3}" \
  --iterations "${ITERATIONS:-5}" \
  --minimum-counter-seconds "${MIN_COUNTER_SECONDS:-3}" \
  --seed 20261006 \
  --result-dir "${RESULT_ROOT}/matrix"

echo "Decode L2 core-only uProf: ${RESULT_ROOT}/matrix"
