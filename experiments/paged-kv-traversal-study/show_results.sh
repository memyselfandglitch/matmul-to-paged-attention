#!/usr/bin/env bash
# Display one Phase 1 + Phase 2 result; default to the newest result.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if (( $# > 1 )); then
  echo "usage: $0 [job-id]" >&2
  exit 1
fi

if (( $# == 1 )); then
  readonly result_dir="results/job-$1"
else
  shopt -s nullglob
  result_dirs=(results/job-*)
  shopt -u nullglob

  if (( ${#result_dirs[@]} == 0 )); then
    echo "error: no results/job-* directory exists yet" >&2
    exit 1
  fi

  result_dir="$(printf '%s\n' "${result_dirs[@]}" | sort -V | tail -n 1)"
  readonly result_dir
fi

if [[ ! -d "${result_dir}" ]]; then
  echo "error: ${result_dir} does not exist" >&2
  exit 1
fi

echo "Showing ${result_dir}"

if [[ -f "${result_dir}/analysis.txt" ]]; then
  echo
  echo "PHASE 1 — fixed BNHD memory"
  sed -n '1,240p' "${result_dir}/analysis.txt"
fi

if [[ -f "${result_dir}/phase2-analysis.txt" ]]; then
  echo
  echo "PHASE 2 — complete layout/traversal matrix"
  sed -n '1,320p' "${result_dir}/phase2-analysis.txt"
else
  echo
  echo "Phase 2 results are not present in ${result_dir}."
  echo "The job may still be running; ./run.sh waits and prints only complete results."
fi
