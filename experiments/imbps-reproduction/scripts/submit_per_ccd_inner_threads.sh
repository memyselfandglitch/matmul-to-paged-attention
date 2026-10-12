#!/usr/bin/env bash

set -euo pipefail

matrix_id="$(sbatch --parsable slurm/per_ccd_inner_threads.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${matrix_id}" \
  --export="ALL,PER_CCD_ARRAY_JOB_ID=${matrix_id},PER_CCD_RESULT_LABEL=per-ccd-inner-threads" \
  slurm/per_ccd_summary.sbatch)"

printf 'per_ccd_inner_threads=%s summary=%s\n' "${matrix_id}" "${summary_id}"
