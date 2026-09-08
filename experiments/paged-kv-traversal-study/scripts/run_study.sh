#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BUILD_DIR="${REPO_ROOT}/build"
readonly RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results}"

readonly BLOCKS="${BLOCKS:-128}"
readonly HEADS="${HEADS:-16}"
readonly BLOCK_SIZE="${BLOCK_SIZE:-16}"
readonly HEAD_DIM="${HEAD_DIM:-64}"
readonly WARMUPS="${WARMUPS:-1}"
readonly REPEATS="${REPEATS:-5}"

if command -v cmake >/dev/null 2>&1; then
  cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${BUILD_DIR}" --parallel
else
  readonly CXX="${CXX:-c++}"
  mkdir -p "${BUILD_DIR}"
  "${CXX}" -std=c++20 -O3 -Wall -Wextra -Wpedantic \
    "${REPO_ROOT}/src/paged_kv_study.cpp" \
    -o "${BUILD_DIR}/paged_kv_study"
fi
mkdir -p "${RESULT_DIR}"

common_args=(
  --blocks "${BLOCKS}"
  --heads "${HEADS}"
  --block-size "${BLOCK_SIZE}"
  --head-dim "${HEAD_DIM}"
  --warmups "${WARMUPS}"
  --repeats "${REPEATS}"
)

echo "Phase 1A: fixed default NHD memory, sequential block table"
"${BUILD_DIR}/paged_kv_study" \
  "${common_args[@]}" \
  --stage fixed \
  --block-order sequential \
  --csv "${RESULT_DIR}/phase1-sequential.csv"

echo
echo "Phase 1B: fixed default NHD memory, shuffled block table"
"${BUILD_DIR}/paged_kv_study" \
  "${common_args[@]}" \
  --stage fixed \
  --block-order shuffled \
  --csv "${RESULT_DIR}/phase1-shuffled.csv"

echo
python3 "${REPO_ROOT}/python/analyze_phase1.py" \
  "${RESULT_DIR}/phase1-sequential.csv" \
  "${RESULT_DIR}/phase1-shuffled.csv" \
  | tee "${RESULT_DIR}/analysis.txt"

if [[ "${RUN_PHASE2:-0}" == "1" ]]; then
  echo
  echo "Phase 2A: full layout/traversal matrix, sequential block table"
  "${BUILD_DIR}/paged_kv_study" \
    "${common_args[@]}" \
    --stage full \
    --block-order sequential \
    --csv "${RESULT_DIR}/phase2-sequential.csv"

  echo
  echo "Phase 2B: full layout/traversal matrix, shuffled block table"
  "${BUILD_DIR}/paged_kv_study" \
    "${common_args[@]}" \
    --stage full \
    --block-order shuffled \
    --csv "${RESULT_DIR}/phase2-shuffled.csv"

  echo
  python3 "${REPO_ROOT}/python/analyze_phase2.py" \
    "${RESULT_DIR}/phase2-sequential.csv" \
    "${RESULT_DIR}/phase2-shuffled.csv" \
    | tee "${RESULT_DIR}/phase2-analysis.txt"
fi

echo
echo "Results: ${RESULT_DIR}"
