#!/usr/bin/env bash

set -euo pipefail

# First verify that PACE v1.0 normal generation accepts batch=2 with both backends.
smoke_id="$(sbatch --parsable \
  --array=4,5%1 \
  --time=01:00:00 \
  --export=ALL,BATCHED_DECODE_CONTEXTS=16,BATCHED_DECODE_OUTPUT_TOKENS=4,BATCHED_DECODE_ROUNDS=1,BATCHED_DECODE_WARMUPS=0 \
  slurm/batched_decode_l2.sbatch)"

# Full 5 batches x 4 backends/K matrix. Each task measures all three contexts.
matrix_id="$(sbatch --parsable \
  --dependency="afterok:${smoke_id}" \
  slurm/batched_decode_l2.sbatch)"

summary_id="$(sbatch --parsable \
  --dependency="afterok:${matrix_id}" \
  --export="ALL,BATCHED_DECODE_ARRAY_JOB_ID=${matrix_id}" \
  slurm/batched_decode_l2_summary.sbatch)"

printf 'batched_decode_smoke=%s matrix=%s summary=%s\n' \
  "${smoke_id}" "${matrix_id}" "${summary_id}"
