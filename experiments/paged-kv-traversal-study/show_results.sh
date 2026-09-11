#!/usr/bin/env bash
# Display the newest Phase 1 + Phase 2 result without requiring a job ID.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

shopt -s nullglob
result_dirs=(results/job-*)
shopt -u nullglob

if (( ${#result_dirs[@]} == 0 )); then
  echo "error: no results/job-* directory exists yet" >&2
  exit 1
fi

latest_dir="$(printf '%s\n' "${result_dirs[@]}" | sort -V | tail -n 1)"
readonly latest_dir

echo "Showing ${latest_dir}"

if [[ -f "${latest_dir}/analysis.txt" ]]; then
  echo
  echo "PHASE 1 — fixed BNHD memory"
  sed -n '1,240p' "${latest_dir}/analysis.txt"
fi

if [[ -f "${latest_dir}/phase2-analysis.txt" ]]; then
  echo
  echo "PHASE 2 — complete layout/traversal matrix"
  sed -n '1,320p' "${latest_dir}/phase2-analysis.txt"
else
  echo
  echo "Phase 2 results are not present in ${latest_dir}."
  echo "Run ./run.sh to submit both phases together."
fi
