#!/usr/bin/env bash

set -euo pipefail

# Static/source-backed private-L2 ownership analysis.
l2_ownership_id="$(sbatch --parsable slurm/l2_ownership.sbatch)"

# Validate the new persistent-pthread path before its full clean-timing matrix.
thread_smoke_id="$(sbatch --parsable \
  --array=0,1,8%1 \
  --time=00:45:00 \
  --export=ALL,PER_CCD_ROWS=128,PER_CCD_ROUNDS=1,PER_CCD_WARMUPS=1 \
  slurm/per_ccd_threads.sbatch)"
thread_matrix_id="$(sbatch --parsable \
  --dependency="afterok:${thread_smoke_id}" \
  slurm/per_ccd_threads.sbatch)"
thread_summary_id="$(sbatch --parsable \
  --dependency="afterok:${thread_matrix_id}" \
  --export="ALL,PER_CCD_ARRAY_JOB_ID=${thread_matrix_id},PER_CCD_RESULT_LABEL=per-ccd-threads" \
  slurm/per_ccd_summary.sbatch)"

# Collect cumulative L2/L3 metrics broken down by CCD only after clean timing.
thread_l3_id="$(sbatch --parsable \
  --dependency="afterok:${thread_matrix_id}" \
  slurm/uprof_per_ccd_threads.sbatch)"
thread_l3_summary_id="$(sbatch --parsable \
  --dependency="afterok:${thread_l3_id}" \
  --export="ALL,PER_CCD_UPROF_ARRAY_JOB_ID=${thread_l3_id}" \
  slurm/uprof_per_ccd_summary.sbatch)"

# Validate both public PACE generation paths before the 16-variant matrix.
full_smoke_id="$(sbatch --parsable \
  --array=0,4%1 \
  --time=01:00:00 \
  --export="ALL,FULL_DECODE_CONTEXTS=16,FULL_DECODE_OUTPUT_TOKENS=4,FULL_DECODE_ROUNDS=1,FULL_DECODE_WARMUPS=0" \
  slurm/full_decode.sbatch)"
full_matrix_id="$(sbatch --parsable \
  --dependency="afterok:${full_smoke_id}" \
  slurm/full_decode.sbatch)"
full_summary_id="$(sbatch --parsable \
  --dependency="afterok:${full_matrix_id}" \
  --export="ALL,FULL_DECODE_ARRAY_JOB_ID=${full_matrix_id}" \
  slurm/full_decode_summary.sbatch)"

printf 'l2_ownership=%s\n' "${l2_ownership_id}"
printf 'thread_smoke=%s thread_matrix=%s thread_summary=%s\n' \
  "${thread_smoke_id}" "${thread_matrix_id}" "${thread_summary_id}"
printf 'thread_l3=%s thread_l3_summary=%s\n' \
  "${thread_l3_id}" "${thread_l3_summary_id}"
printf 'full_decode_smoke=%s full_decode_matrix=%s full_decode_summary=%s\n' \
  "${full_smoke_id}" "${full_matrix_id}" "${full_summary_id}"
