#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-$repo_dir/.venv-pace-v1/bin/python}
uprof_bin=${UPROF_BIN:-/data/scratch/deveshisingh/tools/AMDuProf_Linux_x64_5.3.521/bin/AMDuProfPcm}
config=${UPROF_CONFIG:-$repo_dir/config/uprof_l3_access_miss_zen4.xml}
output_dir=${OUTPUT_DIR:-results/pace-v1-opt30-one-socket-uprof-l3}
splits=${SPLITS:-1,4,8,23}
repetitions=${REPETITIONS:-3}
iterations=${ITERATIONS:-3}
threads=${THREADS:-96}
start_delay_ms=${START_DELAY_MS:-12000}
measurement_delay=${MEASUREMENT_DELAY:-8.0}
uprof_mode=${UPROF_MODE:-perf}

case "$uprof_mode" in
    perf) mode_args=() ;;
    msr) mode_args=(--msr) ;;
    *) echo "UPROF_MODE must be perf or msr" >&2; exit 2 ;;
esac

mkdir -p "$output_dir"
IFS=',' read -r -a split_values <<< "$splits"

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active

for repetition in $(seq 1 "$repetitions"); do
    for split_k in "${split_values[@]}"; do
        if [[ "$split_k" == 1 ]]; then
            label=reference
        else
            label=imbps
        fi
        report="$output_dir/r${repetition}-${label}-k${split_k}.csv"
        echo "[pace-uprof] repetition=$repetition K=$split_k report=$report"
        "$uprof_bin" \
            -i "$config" "${mode_args[@]}" -c package=0 -A package -C -P 6 \
            --start-delay "$start_delay_ms" -q -o "$report" -- \
            numactl --physcpubind=0-95 --membind=0 \
            "$python_bin" scripts/profile_pace_mlp.py \
                --tokens 30720 --hidden-size 7168 --intermediate-size 28672 \
                --dtype bf16 --split-k "$split_k" --threads "$threads" \
                --warmup 1 --iterations "$iterations" \
                --measurement-delay "$measurement_delay"
    done
done

"$python_bin" scripts/summarize_uprof.py "$output_dir"
