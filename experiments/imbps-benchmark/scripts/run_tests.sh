#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

python_bin=${PYTHON_BIN:-python3}
PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m unittest discover -s tests -v
PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m imbps_bench correctness --threads 1

