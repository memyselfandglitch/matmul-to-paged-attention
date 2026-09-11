#!/usr/bin/env bash
# Submit the complete Phase 1 + Phase 2 study from any working directory.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "error: sbatch is unavailable; run this script on the IISc Slurm server" >&2
  exit 1
fi

submission="$(sbatch --parsable --export=ALL,RUN_PHASE2=1 \
  "${REPO_ROOT}/slurm/run_cpu_study.sbatch")"
readonly job_id="${submission%%;*}"

echo "Submitted Phase 1 + Phase 2 as Slurm job ${job_id}."
echo "Monitor: squeue -j ${job_id}"
echo "After it finishes: ./show_results.sh"
echo "Raw results: results/job-${job_id}/"
