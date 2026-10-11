#!/usr/bin/env bash

set -euo pipefail

ccd_id="$(sbatch --parsable slurm/per_ccd_threads.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${ccd_id}" \
  --export="ALL,PER_CCD_ARRAY_JOB_ID=${ccd_id},PER_CCD_RESULT_LABEL=per-ccd-threads" \
  slurm/per_ccd_summary.sbatch)"

printf 'per_ccd_threads=%s summary=%s\n' "${ccd_id}" "${summary_id}"
