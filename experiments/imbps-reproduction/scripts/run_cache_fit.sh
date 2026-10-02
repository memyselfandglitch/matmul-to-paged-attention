#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

readonly CLAIM="${CACHE_FIT_CLAIM:-cache_fit_opt30b}"
case "${CLAIM}" in
  cache_fit_opt30b|cache_resident_control) ;;
  *)
    echo "Unsupported CACHE_FIT_CLAIM: ${CLAIM}" >&2
    exit 2
    ;;
esac

python src/run_standalone_matrix.py \
  --claim "${CLAIM}" \
  --rounds "${ROUNDS:-5}" \
  --warmups "${WARMUPS:-3}" \
  --iterations "${ITERATIONS:-7}"
