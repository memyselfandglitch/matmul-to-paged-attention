#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
pace_dir=${PACE_DIR:-/data/scratch/deveshisingh/AMD-PACE-v1}
python_bin=${PYTHON_BIN:-$repo_dir/.venv-pace-v1/bin/python}
placement=${PLACEMENT:-ccd}
splits=${SPLITS:-1,2,4,8}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
output_dir=${OUTPUT_DIR:-$repo_dir/results/pace-v1-opt125m-e2e-${placement}-${stamp}}

case "$placement" in
    ccd)
        threads=${THREADS:-8}
        numa_args=(--physcpubind=0-7 --membind=0)
        ;;
    one-socket)
        threads=${THREADS:-96}
        numa_args=(--physcpubind=0-95 --membind=0)
        ;;
    two-socket)
        threads=${THREADS:-192}
        numa_args=(--physcpubind=0-191 --interleave=0,1)
        ;;
    *) echo "PLACEMENT must be ccd, one-socket, or two-socket" >&2; exit 2 ;;
esac

mkdir -p "$output_dir" "$pace_dir/benchmarks/llm/performance/benchmark_results/imbps-study"
generated="$pace_dir/benchmarks/llm/performance/benchmark_results/imbps-study/facebook--opt-125m_torch.bfloat16_bs1_it256_nt16_results.json"

export OMP_NUM_THREADS=$threads
export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export TOKENIZERS_PARALLELISM=false

run_config() {
    local label=$1
    local config=$2
    echo "[pace-e2e] placement=$placement threads=$threads variant=$label"
    (
        cd "$pace_dir/benchmarks/llm/performance"
        # PACE's data generator uses Python's random module but does not seed
        # it from generation_args.manual_seed.  Seed before executing the
        # unmodified official entry point so every backend receives the same
        # synthetic token IDs.
        numactl "${numa_args[@]}" "$python_bin" -c \
            'import random, runpy, sys; random.seed(0); sys.argv = ["benchmark_llm_offline.py", "--config", sys.argv[1]]; runpy.run_path("benchmark_llm_offline.py", run_name="__main__")' \
            "$config"
    ) 2>&1 | tee "$output_dir/$label.log"
    cp "$generated" "$output_dir/$label.json"
}

run_config tpp "$repo_dir/config/pace_opt125m_tpp.json"
IFS=',' read -r -a split_values <<< "$splits"
for split_k in "${split_values[@]}"; do
    export IMBPS_BLOCK_SIZE=$split_k
    run_config "imbps-k${split_k}" "$repo_dir/config/pace_opt125m_imbps.json"
done

"$python_bin" - "$output_dir" "$placement" "$threads" <<'PY'
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
payload = {
    "platform": platform.platform(),
    "placement": sys.argv[2],
    "threads": int(sys.argv[3]),
    "affinity_environment": {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY")
    },
    "torch": __import__("torch").__version__,
    "pace_commit": "cfbe8b551cca18c686b771144c18129242796ea1",
    "lscpu": subprocess.run(["lscpu"], text=True, capture_output=True).stdout,
    "numactl": subprocess.run(["numactl", "--hardware"], text=True, capture_output=True).stdout,
}
(output_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
