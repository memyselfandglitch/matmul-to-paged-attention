#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-python3}
model=${MODEL:-facebook/opt-125m}
batch_size=${BATCH_SIZE:-1}
input_tokens=${INPUT_TOKENS:-256}
output_tokens=${OUTPUT_TOKENS:-16}
splits=${SPLITS:-1,2,4,5,6,7,8}
threads=${THREADS:-8}
repeats=${REPEATS:-5}
output_dir=${OUTPUT_DIR:-results/hf-opt-e2e-zen4}
export HF_HOME=${HF_HOME:-$repo_dir/.cache/huggingface}

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=close
export OMP_PLACES=cores

exec numactl --physcpubind=0-7 --membind=0 \
    "$python_bin" -m imbps_bench hf-opt-e2e \
    --model-name-or-path "$model" \
    --batch-size "$batch_size" \
    --input-tokens "$input_tokens" \
    --output-tokens "$output_tokens" \
    --dtype bf16 \
    --splits "$splits" \
    --threads "$threads" \
    --warmup 1 \
    --repeats "$repeats" \
    --weight-layout prepacked \
    --attn-implementation sdpa \
    --output-dir "$output_dir"
