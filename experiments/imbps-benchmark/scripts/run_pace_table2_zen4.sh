#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-$repo_dir/.venv-pace-v1/bin/python}
placement=${PLACEMENT:-one-socket}
tokens=${TOKENS:-30720}
hidden_size=${HIDDEN_SIZE:-7168}
intermediate_size=${INTERMEDIATE_SIZE:-28672}
dtype=${DTYPE:-bf16}
splits=${SPLITS:-1,4,8,16,23}
warmup=${WARMUP:-2}
repeats=${REPEATS:-10}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
output_dir=${OUTPUT_DIR:-results/pace-v1-opt30-${placement}-${stamp}}

case "$placement" in
    one-socket)
        threads=${THREADS:-96}
        cpu_list=${CPU_LIST:-0-95}
        numa_args=(--physcpubind="$cpu_list" --membind=0)
        ;;
    two-socket)
        threads=${THREADS:-192}
        cpu_list=${CPU_LIST:-0-191}
        numa_args=(--physcpubind="$cpu_list" --interleave=0,1)
        ;;
    ccd)
        threads=${THREADS:-8}
        cpu_list=${CPU_LIST:-0-7}
        numa_args=(--physcpubind="$cpu_list" --membind=0)
        ;;
    *)
        echo "PLACEMENT must be one-socket, two-socket, or ccd" >&2
        exit 2
        ;;
esac

if [[ ! -x "$python_bin" ]]; then
    echo "PACE Python not found at $python_bin" >&2
    exit 1
fi

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export PACE_GIT_COMMIT=${PACE_GIT_COMMIT:-cfbe8b551cca18c686b771144c18129242796ea1}

runner=("$python_bin" scripts/benchmark_pace_mlp.py
    --tokens "$tokens"
    --hidden-size "$hidden_size"
    --intermediate-size "$intermediate_size"
    --dtype "$dtype"
    --splits "$splits"
    --threads "$threads"
    --warmup "$warmup"
    --repeats "$repeats"
    --output-dir "$output_dir")

echo "[pace] placement=$placement CPUs=$cpu_list threads=$threads output=$output_dir"
if command -v numactl >/dev/null 2>&1; then
    numactl "${numa_args[@]}" "${runner[@]}"
else
    echo "numactl is required for controlled placement" >&2
    exit 1
fi
