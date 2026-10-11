#!/usr/bin/env bash

set -euo pipefail

matrix_id="$(sbatch --parsable slurm/uprof_per_ccd_threads.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${matrix_id}" \
  --export="ALL,PER_CCD_UPROF_ARRAY_JOB_ID=${matrix_id}" \
  slurm/uprof_per_ccd_summary.sbatch)"

printf 'per_ccd_l3=%s summary=%s\n' "${matrix_id}" "${summary_id}"
