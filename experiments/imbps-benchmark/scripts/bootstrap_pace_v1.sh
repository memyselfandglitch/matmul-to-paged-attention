#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv_dir=${PACE_VENV:-$repo_dir/.venv-pace-v1}
pace_src=${PACE_SRC:-$repo_dir/.vendor/AMD-PACE-v1}
python_bin=${PYTHON_BIN:-python3}

if [[ ! -d "$pace_src/.git" ]]; then
    mkdir -p "$(dirname "$pace_src")"
    git clone --branch v1.0 --depth 1 https://github.com/amd/AMD-PACE.git "$pace_src"
fi

"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r "$pace_src/build_requirements.txt"
"$venv_dir/bin/python" -m pip install 'numpy>=2.0.2' hypothesis

(
    cd "$pace_src"
    MAX_JOBS=${MAX_JOBS:-32} CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-32} \
        "$venv_dir/bin/python" -m pip install --no-build-isolation -v .
)

# PACE v1 imports its LLM and visualization modules from package __init__, so
# the standalone fused operator still needs these runtime dependencies.
"$venv_dir/bin/python" -m pip install \
    transformers==4.51.3 \
    'huggingface-hub[hf_xet]==0.31.4' \
    'safetensors>=0.5.2' \
    'psutil>=7.0.0' \
    matplotlib==3.10.7

"$venv_dir/bin/python" -c \
    'import pace, torch; print("torch", torch.__version__); print("PACE MLP registered", hasattr(torch.ops.pace, "mlp_mlp_fusion"))'
