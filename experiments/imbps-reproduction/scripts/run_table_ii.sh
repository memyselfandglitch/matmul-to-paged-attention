#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python3 src/run_standalone_matrix.py \
  --claim table_ii \
  --rounds "${ROUNDS:-5}" \
  --warmups "${WARMUPS:-3}" \
  --iterations "${ITERATIONS:-7}"
