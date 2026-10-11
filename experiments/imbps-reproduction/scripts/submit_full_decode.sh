#!/usr/bin/env bash

set -euo pipefail

decode_id="$(sbatch --parsable slurm/full_decode.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${decode_id}" \
  --export="ALL,FULL_DECODE_ARRAY_JOB_ID=${decode_id}" \
  slurm/full_decode_summary.sbatch)"

printf 'full_decode=%s summary=%s\n' "${decode_id}" "${summary_id}"
