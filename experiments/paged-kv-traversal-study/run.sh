#!/usr/bin/env bash
# Submit, wait for, and display the complete layout/traversal study.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "error: sbatch is unavailable; run this script on the IISc Slurm server" >&2
  exit 1
fi

submission="$(sbatch --parsable --export=ALL,RUN_PHASE2=1,RUN_CROSSOVER=1 \
  "${REPO_ROOT}/slurm/run_cpu_study.sbatch")"
readonly job_id="${submission%%;*}"

echo "Submitted Phase 1 + Phase 2 + fragmentation crossover as Slurm job ${job_id}."
echo "Raw results: results/job-${job_id}/"

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

# Slurm may remove a completed job from squeue just before its files become
# visible. Give the filesystem a few seconds to settle.
for _ in {1..10}; do
  if [[ -f "results/job-${job_id}/phase2-analysis.txt" && \
        -f "results/job-${job_id}/crossover-analysis.txt" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "results/job-${job_id}/phase2-analysis.txt" || \
      ! -f "results/job-${job_id}/crossover-analysis.txt" ]]; then
  echo "error: job ${job_id} ended without complete Phase 2 and crossover reports" >&2
  sacct --jobs "${job_id}" --format=JobID,State,ExitCode 2>/dev/null || true
  echo "Inspect cpu-study-${job_id}.out and cpu-study-${job_id}.err." >&2
  exit 1
fi

echo "Job ${job_id}: COMPLETED"
"${REPO_ROOT}/show_results.sh" "${job_id}"
