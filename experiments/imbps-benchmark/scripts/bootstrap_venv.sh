#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

system_python=${SYSTEM_PYTHON:-python3}
venv_dir=${VENV_DIR:-$repo_dir/.venv}
scratch_cache=${PIP_CACHE_DIR:-$repo_dir/.cache/pip}
scratch_tmp=${TMPDIR:-$repo_dir/.tmp}
torch_index_url=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}

mkdir -p "$scratch_cache" "$scratch_tmp"
export PIP_CACHE_DIR=$scratch_cache
export TMPDIR=$scratch_tmp

if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$system_python" -m venv "$venv_dir"
fi

venv_python=$venv_dir/bin/python
"$venv_python" -m pip install --upgrade pip setuptools wheel

if ! "$venv_python" -c 'import torch' >/dev/null 2>&1; then
    "$venv_python" -m pip install --index-url "$torch_index_url" 'torch>=2.3'
fi

"$venv_python" -m pip install -e . --no-deps

echo
echo "Environment ready: $venv_dir"
echo "Activate it with: source $venv_dir/bin/activate"
"$venv_python" -c 'import torch; print("PyTorch", torch.__version__); print(torch.__config__.show())'

