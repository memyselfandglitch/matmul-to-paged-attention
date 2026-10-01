#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python3 -m compileall -q src tests
python3 -m unittest discover -s tests -v

for script in scripts/*.sh slurm/*.sbatch; do
  bash -n "${script}"
done

for program in src/*.py; do
  python3 "${program}" --help >/dev/null
done

echo "Harness verification passed"
