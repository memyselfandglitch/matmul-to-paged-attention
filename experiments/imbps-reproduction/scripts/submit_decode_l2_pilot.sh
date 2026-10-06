#!/usr/bin/env bash
set -euo pipefail

timing_id="$(sbatch --parsable slurm/decode_l2_pilot.sbatch)"
uprof_id="$(sbatch --parsable \
  --dependency="afterok:${timing_id}" \
  slurm/uprof_decode_l2_pilot.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${uprof_id}" \
  --export="ALL,UPROF_ARRAY_JOB_ID=${uprof_id},UPROF_RESULT_LABEL=decode-l2-pilot" \
  slurm/uprof_summarize.sbatch)"

printf 'timing=%s uprof=%s summary=%s\n' \
  "${timing_id}" "${uprof_id}" "${summary_id}"
