#!/usr/bin/env bash

set -euo pipefail

matrix_id="$(sbatch --parsable slurm/uprof_per_ccd_optimized.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${matrix_id}" \
  --export="ALL,PER_CCD_UPROF_ARRAY_JOB_ID=${matrix_id},PER_CCD_UPROF_RESULT_LABEL=uprof-per-ccd-optimized" \
  slurm/uprof_per_ccd_summary.sbatch)"

printf 'per_ccd_optimized_l3=%s summary=%s\n' "${matrix_id}" "${summary_id}"
