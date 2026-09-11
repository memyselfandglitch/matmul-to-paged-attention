#!/usr/bin/env bash
# Submit, wait for, and display the complete matrix-multiplication study.

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
readonly result_dir="${STUDY_ROOT}/results/job-${job_id}"

echo "Submitted optimization-stage and final loop-order studies as job ${job_id}."
echo "Quick presentation sweep: n=384, 768, 1152 with 3 timed repetitions."

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
  if [[ -f "${result_dir}/loop-order.txt" && \
        -f "${result_dir}/optimization-stages.txt" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "${result_dir}/loop-order.txt" || \
      ! -f "${result_dir}/optimization-stages.txt" ]]; then
  echo "error: job ${job_id} ended without both result files" >&2
  sacct --jobs "${job_id}" --format=JobID,State,ExitCode 2>/dev/null || true
  echo "Inspect loop-order-${job_id}.out and loop-order-${job_id}.err." >&2
  exit 1
fi

echo
echo "PART 1 — MATRIX-MULTIPLICATION OPTIMIZATION STAGES"
sed -n '1,320p' "${result_dir}/optimization-stages.txt"

echo
echo "PART 2 — ALL LOOP ORDERS USING THE FINAL OPTIMIZATION STACK"
echo "Each order uses cache tiling + SIMD register blocking; only Mc/Nc/Kc order changes."
sed -n '1,320p' "${result_dir}/loop-order.txt"

echo
echo "Raw results: ${result_dir}"
