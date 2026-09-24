#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

splits=${SPLITS:-1,2,3,4,5,6,7,8,12,14,16,23}
repeats=${REPEATS:-30}
warmup=${WARMUP:-3}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
placements=${PLACEMENTS:-one-socket,two-socket}

IFS=, read -r -a placement_list <<< "$placements"
for placement in "${placement_list[@]}"; do
    output_dir=${OUTPUT_ROOT:-results/pace-v1-k-validation-${stamp}}/$placement
    echo "[pace-validation] placement=$placement output=$output_dir"
    PLACEMENT="$placement" \
    SPLITS="$splits" \
    REPEATS="$repeats" \
    WARMUP="$warmup" \
    OUTPUT_DIR="$output_dir" \
        ./scripts/run_pace_table2_zen4.sh
    PYTHONPATH="$repo_dir" "${PYTHON_BIN:-$repo_dir/.venv-pace-v1/bin/python}" \
        scripts/analyze_pace_k_sweep.py \
        --raw "$output_dir/raw.csv" \
        --output-csv "$output_dir/validation.csv" \
        --output-json "$output_dir/validation.json"
done
