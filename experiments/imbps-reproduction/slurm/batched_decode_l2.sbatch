#!/usr/bin/env bash
#SBATCH --job-name=imbps-batched-decode
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --array=0-19%1
#SBATCH --output=batched-decode-l2-%A_%a.out
#SBATCH --error=batched-decode-l2-%A_%a.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly THREADS="${SLURM_CPUS_PER_TASK}"
readonly CLAIM="${BATCHED_DECODE_CLAIM:-qwen25_7b_batched_normal_l2}"
readonly BATCHES=(1 2 4 8 16)
readonly BACKENDS=(tpp imbps imbps imbps)
readonly SPLITS=(1 2 4 8)
readonly VARIANT_COUNT="${#BACKENDS[@]}"
readonly BATCH_INDEX="$((SLURM_ARRAY_TASK_ID / VARIANT_COUNT))"
readonly VARIANT_INDEX="$((SLURM_ARRAY_TASK_ID % VARIANT_COUNT))"
readonly BATCH="${BATCHES[${BATCH_INDEX}]}"
readonly BACKEND="${BACKENDS[${VARIANT_INDEX}]}"
readonly SPLIT="${SPLITS[${VARIANT_INDEX}]}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/batched-decode-l2-${SLURM_ARRAY_JOB_ID}"
readonly CASE_DIR="${RESULT_ROOT}/b${BATCH}-${BACKEND}-k${SPLIT}"

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
if [[ -n "${BATCHED_DECODE_CONTEXTS:-}" ]]; then
  optional_args+=(--contexts "${BATCHED_DECODE_CONTEXTS}")
fi
if [[ -n "${BATCHED_DECODE_OUTPUT_TOKENS:-}" ]]; then
  optional_args+=(--output-tokens "${BATCHED_DECODE_OUTPUT_TOKENS}")
fi
if [[ -n "${BATCHED_DECODE_TARGET_MODEL:-}" ]]; then
  optional_args+=(--target-model "${BATCHED_DECODE_TARGET_MODEL}")
fi

numactl --physcpubind="${CPU_LIST}" --membind=0 \
  python3 src/full_decode_bench.py \
    --claim "${CLAIM}" \
    --mode normal \
    --backend "${BACKEND}" \
    --splits "${SPLIT}" \
    --draft-tokens 0 \
    --batch "${BATCH}" \
    --rounds "${BATCHED_DECODE_ROUNDS:-3}" \
    --warmups "${BATCHED_DECODE_WARMUPS:-1}" \
    --threads "${THREADS}" \
    "${optional_args[@]}" \
    --output "${CASE_DIR}/result.json"

echo "Batched decode result: ${CASE_DIR}/result.json"
