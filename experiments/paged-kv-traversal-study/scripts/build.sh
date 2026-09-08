#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BUILD_DIR="${REPO_ROOT}/build"

if command -v cmake >/dev/null 2>&1; then
  cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${BUILD_DIR}" --parallel
else
  readonly CXX="${CXX:-c++}"
  mkdir -p "${BUILD_DIR}"
  "${CXX}" -std=c++20 -O3 -march=native -Wall -Wextra -Wpedantic \
    "${REPO_ROOT}/src/paged_kv_study.cpp" \
    -o "${BUILD_DIR}/paged_kv_study"
fi
