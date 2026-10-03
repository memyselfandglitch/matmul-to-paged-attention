#!/usr/bin/env bash
#SBATCH --job-name=imbps-upr-pilot
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=uprof-counter-pilot-%j.out
#SBATCH --error=uprof-counter-pilot-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/uprof-counter-pilot-${RUN_STAMP}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

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

readonly THREADS=96
readonly CPU_LIST="$(python src/select_llc_cpus.py --threads "${THREADS}")"
export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=ACTIVE
export LIBXSMM_BLOCK_SIZE=32
export EXPECTED_PHYSICAL_CORES="${SLURM_CPUS_PER_TASK}"
export REQUIRE_SINGLE_SOCKET=1
export REJECT_SMT=1

numactl --cpunodebind=0 --membind=0 ./scripts/preflight.sh
mkdir -p "${RESULT_ROOT}"
{
  echo "uprof_binary=${UPROF_BIN}"
  echo "counter_scope=package=0"
  echo "cpu_list=${CPU_LIST}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
} > "${RESULT_ROOT}/pilot-environment.txt"

for pass in cache traffic; do
  if [[ "${pass}" == cache ]]; then
    metrics="ipc,l2,l3"
  else
    metrics="memory"
  fi
  for variant in tpp:1 imbps:4; do
    backend="${variant%%:*}"
    split="${variant##*:}"
    python src/run_uprof_case.py \
      --uprof-bin "${UPROF_BIN}" \
      --access-mode perf \
      --metrics "${metrics}" \
      --package 0 \
      --cpu-list "${CPU_LIST}" \
      --backend "${backend}" \
      --split "${split}" \
      --batch 16 \
      --sequence 1920 \
      --threads "${THREADS}" \
      --warmups 3 \
      --iterations 3 \
      --result-dir "${RESULT_ROOT}/${pass}-${backend}-k${split}"
  done
done

for csv in "${RESULT_ROOT}"/*/uprof.csv; do
  echo "===== ${csv} ====="
  head -n 40 "${csv}"
done
for log in "${RESULT_ROOT}"/*/uprof.log; do
  echo "===== warnings: ${log} ====="
  grep -iE 'warning|error|multiplex|package|socket' "${log}" || true
done
echo "uProf counter pilot: ${RESULT_ROOT}"
