#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-python3}
mlp_kind=${MLP_KIND:-opt}
tokens=${TOKENS:-256}
hidden_size=${HIDDEN_SIZE:-4096}
intermediate_size=${INTERMEDIATE_SIZE:-16384}
dtype=${DTYPE:-bf16}
splits=${SPLITS:-1,2,4,5,6,7,8,16,32,64}
threads=${THREADS:-8}
warmup=${WARMUP:-3}
repeats=${REPEATS:-10}
weight_layout=${WEIGHT_LAYOUT:-prepacked}
cpu_list=${CPU_LIST:-0-7}
numa_node=${NUMA_NODE:-0}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
output_dir=${OUTPUT_DIR:-results/zen4-ccd-${mlp_kind}-${stamp}}

runner=("$python_bin" -m imbps_bench sweep
    --mlp-kind "$mlp_kind"
    --tokens "$tokens"
    --hidden-size "$hidden_size"
    --intermediate-size "$intermediate_size"
    --dtype "$dtype"
    --splits "$splits"
    --threads "$threads"
    --warmup "$warmup"
    --repeats "$repeats"
    --weight-layout "$weight_layout"
    --output-dir "$output_dir")

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=close
export OMP_PLACES=cores

if command -v numactl >/dev/null 2>&1; then
    numactl --physcpubind="$cpu_list" --membind="$numa_node" "${runner[@]}"
elif command -v taskset >/dev/null 2>&1; then
    echo "warning: numactl is unavailable; CPU affinity is set but memory is not bound" >&2
    taskset -c "$cpu_list" "${runner[@]}"
else
    echo "warning: neither numactl nor taskset is available; run is not pinned" >&2
    "${runner[@]}"
fi

echo "Results: $output_dir"
