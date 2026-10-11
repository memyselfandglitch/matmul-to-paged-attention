#!/usr/bin/env bash
#SBATCH --job-name=imbps-full-decode
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --array=0-15%1
#SBATCH --output=full-decode-%A_%a.out
#SBATCH --error=full-decode-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly THREADS="${SLURM_CPUS_PER_TASK}"
readonly CLAIM="${FULL_DECODE_CLAIM:-qwen25_7b_pard}"
readonly MODES=(normal normal normal normal pard pard pard pard pard pard pard pard pard pard pard pard)
readonly BACKENDS=(tpp imbps imbps imbps tpp tpp tpp imbps imbps imbps imbps imbps imbps imbps imbps imbps)
readonly SPLITS=(1 2 4 8 1 1 1 2 2 2 4 4 4 8 8 8)
readonly DRAFTS=(0 0 0 0 4 8 12 4 8 12 4 8 12 4 8 12)
readonly MODE="${MODES[${SLURM_ARRAY_TASK_ID}]}"
readonly BACKEND="${BACKENDS[${SLURM_ARRAY_TASK_ID}]}"
readonly SPLIT="${SPLITS[${SLURM_ARRAY_TASK_ID}]}"
readonly DRAFT="${DRAFTS[${SLURM_ARRAY_TASK_ID}]}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/full-decode-${SLURM_ARRAY_JOB_ID}"
readonly CASE_DIR="${RESULT_ROOT}/${MODE}-d${DRAFT}-${BACKEND}-k${SPLIT}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
source ./scripts/enable_tcmalloc.sh

export HF_HOME="${HF_HOME:-/data/scratch/${USER}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}" "${CASE_DIR}"

readonly CPU_LIST="$(python3 src/select_llc_cpus.py --threads "${THREADS}")"
export OMP_NUM_THREADS="${THREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=ACTIVE
export NUMEXPR_MAX_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export LIBXSMM_BLOCK_SIZE=32

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/validate_runtime.py \
    --require-tcmalloc \
    --require-thp "${REQUIRE_THP_MODE:-always}" \
    --expected-l2-mib 96 \
    --expected-l3-mib 384 \
    --output "${CASE_DIR}/runtime-validation.json"

python3 src/select_llc_cpus.py --threads "${THREADS}" --json \
  > "${CASE_DIR}/cpu-selection.json"

optional_args=()
if [[ -n "${FULL_DECODE_CONTEXTS:-}" ]]; then
  optional_args+=(--contexts "${FULL_DECODE_CONTEXTS}")
fi
if [[ -n "${FULL_DECODE_OUTPUT_TOKENS:-}" ]]; then
  optional_args+=(--output-tokens "${FULL_DECODE_OUTPUT_TOKENS}")
fi
if [[ -n "${FULL_DECODE_TARGET_MODEL:-}" ]]; then
  optional_args+=(--target-model "${FULL_DECODE_TARGET_MODEL}")
fi
if [[ -n "${FULL_DECODE_DRAFT_MODEL:-}" ]]; then
  optional_args+=(--draft-model "${FULL_DECODE_DRAFT_MODEL}")
fi

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/full_decode_bench.py \
    --claim "${CLAIM}" \
    --mode "${MODE}" \
    --backend "${BACKEND}" \
    --splits "${SPLIT}" \
    --draft-tokens "${DRAFT}" \
    --rounds "${FULL_DECODE_ROUNDS:-3}" \
    --warmups "${FULL_DECODE_WARMUPS:-1}" \
    --threads "${THREADS}" \
    "${optional_args[@]}" \
    --output "${CASE_DIR}/result.json"

echo "Full decode result: ${CASE_DIR}/result.json"
