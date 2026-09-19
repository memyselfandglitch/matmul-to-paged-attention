#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

"$repo_dir/scripts/bootstrap_venv.sh"
venv_python=${VENV_DIR:-$repo_dir/.venv}/bin/python
"$venv_python" -m pip install -r "$repo_dir/requirements-hf.txt"
"$venv_python" -m pip install -e "$repo_dir" --no-deps

echo
"$venv_python" -c 'import torch, transformers; print("PyTorch", torch.__version__); print("Transformers", transformers.__version__)'
