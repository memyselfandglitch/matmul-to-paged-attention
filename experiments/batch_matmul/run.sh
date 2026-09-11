#!/usr/bin/env bash
# Submit, wait for, and display the PyTorch matmul/BMM study.

set -euo pipefail

readonly STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${STUDY_ROOT}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "error: sbatch is unavailable; run this script on the IISc Slurm server" >&2
  exit 1
fi

submission="$(sbatch --parsable \
  --export=ALL,STUDY_ROOT="${STUDY_ROOT}" \
  "${STUDY_ROOT}/slurm/run_study.sbatch")"
readonly job_id="${submission%%;*}"
readonly result_file="${STUDY_ROOT}/results/job-${job_id}/bmm-study.txt"
readonly csv_file="${STUDY_ROOT}/results/job-${job_id}/cache-sweep.csv"

echo "Submitted matmul/BMM study as job ${job_id}."

last_state=""
while true; do
  state="$(squeue --noheader --jobs="${job_id}" --format='%T' 2>/dev/null \
    | sed -n '1p' | tr -d '[:space:]')"
  if [[ -z "${state}" ]]; then
    break
  fi
  if [[ "${state}" != "${last_state}" ]]; then
    echo "Job ${job_id}: ${state}"
    last_state="${state}"
  fi
  sleep 3
done

for _ in {1..10}; do
  if [[ -f "${result_file}" && -f "${csv_file}" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "${result_file}" || ! -f "${csv_file}" ]]; then
  echo "error: job ${job_id} ended without both result files" >&2
  sacct --jobs "${job_id}" --format=JobID,State,ExitCode 2>/dev/null || true
  echo "Inspect batch-matmul-${job_id}.out and batch-matmul-${job_id}.err." >&2
  exit 1
fi

echo
echo "PYTORCH MATMUL/BMM STUDY"
sed -n '1,320p' "${result_file}"

echo
echo "Raw results: $(dirname "${result_file}")"
