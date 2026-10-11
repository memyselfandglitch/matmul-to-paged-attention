#!/usr/bin/env bash
#SBATCH --job-name=imbps-per-ccd
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --array=0-10%1
#SBATCH --output=per-ccd-%A_%a.out
#SBATCH --error=per-ccd-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly THREADS="${SLURM_CPUS_PER_TASK}"
readonly CLAIM="${PER_CCD_CLAIM:-opt30b_parallel_imbps}"
readonly VARIANTS=(
  tpp-all96
  global-k12-all96
  global-k24-all96
  global-k48-all96
  parallel-p1-k1
  parallel-p2-k2
  parallel-p4-k4
  parallel-p6-k6
  parallel-p12-k12
  parallel-p12-k24
  parallel-p12-k48
)
readonly VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID}]}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/per-ccd-${SLURM_ARRAY_JOB_ID}"
readonly CASE_DIR="${RESULT_ROOT}/${VARIANT}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
source ./scripts/enable_tcmalloc.sh

readonly CPU_LIST="$(python3 src/select_llc_cpus.py \
  --threads "${THREADS}" \
  --expected-groups 12 \
  --expected-group-size 8)"
export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=ACTIVE
export NUMEXPR_MAX_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export LIBXSMM_BLOCK_SIZE=32

mkdir -p "${CASE_DIR}"
numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/validate_runtime.py \
    --require-tcmalloc \
    --require-thp "${REQUIRE_THP_MODE:-always}" \
    --expected-l2-mib 96 \
    --expected-l3-mib 384 \
    --output "${CASE_DIR}/runtime-validation.json"

python3 src/select_llc_cpus.py \
  --threads "${THREADS}" \
  --expected-groups 12 \
  --expected-group-size 8 \
  --json \
  > "${CASE_DIR}/cpu-selection.json"

optional_args=()
if [[ -n "${PER_CCD_ROWS:-}" ]]; then
  optional_args+=(--rows "${PER_CCD_ROWS}")
fi

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/per_ccd_mlp_bench.py \
    --claim "${CLAIM}" \
    --variant "${VARIANT}" \
    --rounds "${PER_CCD_ROUNDS:-7}" \
    --warmups "${PER_CCD_WARMUPS:-3}" \
    --threads "${THREADS}" \
    "${optional_args[@]}" \
    --output "${CASE_DIR}/result.json"

echo "Per-CCD result: ${CASE_DIR}/result.json"
