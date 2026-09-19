#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-python3}
model=${MODEL:-facebook/opt-125m}
batch_size=${BATCH_SIZE:-1}
sequence_length=${SEQUENCE_LENGTH:-256}
splits=${SPLITS:-1,2,4,5,6,7,8,16}
threads=${THREADS:-8}
repeats=${REPEATS:-10}
output_dir=${OUTPUT_DIR:-results/hf-opt-layer-zen4}
accumulation_dtype=${ACCUMULATION_DTYPE:-input}
export HF_HOME=${HF_HOME:-$repo_dir/.cache/huggingface}

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=close
export OMP_PLACES=cores

exec numactl --physcpubind=0-7 --membind=0 \
    "$python_bin" -m imbps_bench hf-opt-layer \
    --model-name-or-path "$model" \
    --model-mode pretrained \
    --input-source captured \
    --batch-size "$batch_size" \
    --sequence-length "$sequence_length" \
    --dtype bf16 \
    --splits "$splits" \
    --threads "$threads" \
    --warmup 3 \
    --repeats "$repeats" \
    --weight-layout prepacked \
    --accumulation-dtype "$accumulation_dtype" \
    --attn-implementation sdpa \
    --output-dir "$output_dir"
