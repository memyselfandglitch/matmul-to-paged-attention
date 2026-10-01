#!/usr/bin/env bash
#SBATCH --job-name=imbps-uprof-check
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --output=uprof-inventory-%j.out
#SBATCH --error=uprof-inventory-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RESULT_DIR="${PROJECT_ROOT}/results/uprof-inventory-${RUN_STAMP}"
mkdir -p "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

{
  echo "hostname=$(hostname)"
  echo "captured_at_utc=${RUN_STAMP}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_partition=${SLURM_JOB_PARTITION}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK}"
  id
  uname -a
  command -v AMDuProfPcm || true
  command -v perf || true
  cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || true
  ls -l /dev/cpu/0/msr 2>/dev/null || true
} > "${RESULT_DIR}/environment.txt" 2>&1

declare -a UPROF_CANDIDATES=()
if [[ -n "${AMDUPROFPCM_BIN:-}" && -x "${AMDUPROFPCM_BIN}" ]]; then
  UPROF_CANDIDATES+=("${AMDUPROFPCM_BIN}")
fi
if command -v AMDuProfPcm >/dev/null 2>&1; then
  UPROF_CANDIDATES+=("$(command -v AMDuProfPcm)")
fi

declare -a SEARCH_ROOTS=()
for root in /opt /usr/local /apps /software "${HOME:-}" \
  "/data/scratch/${USER:-deveshisingh}"; do
  if [[ -d "${root}" ]]; then
    SEARCH_ROOTS+=("${root}")
  fi
done
if (( ${#SEARCH_ROOTS[@]} > 0 )); then
  while IFS= read -r candidate; do
    UPROF_CANDIDATES+=("${candidate}")
  done < <(
    find "${SEARCH_ROOTS[@]}" -type f -name AMDuProfPcm -perm -u+x \
      2>/dev/null | sort -u
  )
fi

printf '%s\n' "${UPROF_CANDIDATES[@]}" | awk 'NF && !seen[$0]++' \
  > "${RESULT_DIR}/uprof-candidates.txt"
mapfile -t UPROF_CANDIDATES < "${RESULT_DIR}/uprof-candidates.txt"
if (( ${#UPROF_CANDIDATES[@]} == 0 )); then
  echo "AMDuProfPcm was not found on PATH or under the searched roots" >&2
  echo "Inventory: ${RESULT_DIR}"
  exit 1
fi

readonly UPROF_BIN="${UPROF_CANDIDATES[0]}"
echo "selected_uprof_binary=${UPROF_BIN}" >> "${RESULT_DIR}/environment.txt"

"${UPROF_BIN}" -v > "${RESULT_DIR}/uprof-version.txt" 2>&1 || true
"${UPROF_BIN}" -h > "${RESULT_DIR}/uprof-help.txt" 2>&1 || true
"${UPROF_BIN}" -l > "${RESULT_DIR}/uprof-events.txt" 2>&1 || true
perf list > "${RESULT_DIR}/perf-list.txt" 2>&1 || true

set +e
"${UPROF_BIN}" --msr -m ipc,l2,l3,memory -a -C \
  -o "${RESULT_DIR}/permission-probe.csv" -- /bin/sleep 2 \
  > "${RESULT_DIR}/permission-probe.log" 2>&1
readonly PROBE_STATUS=$?
set -e

echo "${PROBE_STATUS}" > "${RESULT_DIR}/permission-probe.exit-code"
echo "uProf permission probe exit code: ${PROBE_STATUS}"
echo "Inventory: ${RESULT_DIR}"
exit "${PROBE_STATUS}"
