#!/usr/bin/env bash
#SBATCH --job-name=imbps-target-verify
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=384G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --array=0-3%1
#SBATCH --output=target-verify-%A_%a.out
#SBATCH --error=target-verify-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly THREADS="${SLURM_CPUS_PER_TASK}"
readonly CLAIM="${TARGET_VERIFY_CLAIM:-target_verify_opt30b_pilot}"
readonly MODEL_REFERENCE="${TARGET_MODEL_REFERENCE:-}"
readonly BACKENDS=(tpp imbps imbps imbps)
readonly SPLITS=(1 2 4 8)
readonly BACKEND="${BACKENDS[${SLURM_ARRAY_TASK_ID}]}"
readonly SPLIT="${SPLITS[${SLURM_ARRAY_TASK_ID}]}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/target-verify-${SLURM_ARRAY_JOB_ID}"
readonly CASE_DIR="${RESULT_ROOT}/${BACKEND}-k${SPLIT}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
source ./scripts/enable_tcmalloc.sh

# OPT-30B is larger than the home quota on mn01. Keep model downloads and
# Hugging Face locks on scratch unless the submitter explicitly overrides them.
export HF_HOME="${HF_HOME:-/data/scratch/${USER}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

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

mkdir -p "${CASE_DIR}"
numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/validate_runtime.py \
    --require-tcmalloc \
    --require-thp "${REQUIRE_THP_MODE:-always}" \
    --expected-l2-mib 96 \
    --output "${CASE_DIR}/runtime-validation.json"

python3 src/select_llc_cpus.py --threads "${THREADS}" --json \
  > "${CASE_DIR}/cpu-selection.json"

model_args=()
if [[ -n "${MODEL_REFERENCE}" ]]; then
  model_args+=(--model "${MODEL_REFERENCE}")
fi

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/pace_target_verify_bench.py \
    --claim "${CLAIM}" \
    --backend "${BACKEND}" \
    --splits "${SPLIT}" \
    --rounds "${TARGET_VERIFY_ROUNDS:-5}" \
    --warmups "${TARGET_VERIFY_WARMUPS:-2}" \
    --iterations "${TARGET_VERIFY_ITERATIONS:-3}" \
    --min-round-seconds "${TARGET_VERIFY_MIN_ROUND_SECONDS:-1}" \
    --threads "${THREADS}" \
    "${model_args[@]}" \
    --output "${CASE_DIR}/result.json"

echo "Target verification result: ${CASE_DIR}/result.json"
