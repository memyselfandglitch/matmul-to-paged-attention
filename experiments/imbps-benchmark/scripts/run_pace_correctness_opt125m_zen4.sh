#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
pace_dir=${PACE_MAIN_DIR:-/data/scratch/deveshisingh/AMD-PACE}
python_bin=${PACE_MAIN_PYTHON:-/data/scratch/deveshisingh/pace-venv/bin/python}
splits=${SPLITS:-1,2,4,8}
threads=${THREADS:-8}
output_dir=${OUTPUT_DIR:-$repo_dir/results/pace-main-opt125m-correctness}
mkdir -p "$output_dir"

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export TOKENIZERS_PARALLELISM=false

run_check() {
    local label=$1
    local config=$2
    echo "[pace-correctness] variant=$label threads=$threads"
    (
        cd "$pace_dir/benchmarks/llm/correctness"
        numactl --physcpubind=0-7 --membind=0 \
            "$python_bin" test_correctness.py --config "$config"
    ) 2>&1 | tee "$output_dir/$label.log"
}

run_check tpp "$repo_dir/config/pace_correctness_opt125m_tpp.json"
IFS=',' read -r -a split_values <<< "$splits"
for split_k in "${split_values[@]}"; do
    export IMBPS_BLOCK_SIZE=$split_k
    run_check "imbps-k${split_k}" "$repo_dir/config/pace_correctness_opt125m_imbps.json"
done
