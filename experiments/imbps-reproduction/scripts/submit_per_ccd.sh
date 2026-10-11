#!/usr/bin/env bash

set -euo pipefail

ccd_id="$(sbatch --parsable slurm/per_ccd.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${ccd_id}" \
  --export="ALL,PER_CCD_ARRAY_JOB_ID=${ccd_id}" \
  slurm/per_ccd_summary.sbatch)"

printf 'per_ccd=%s summary=%s\n' "${ccd_id}" "${summary_id}"
