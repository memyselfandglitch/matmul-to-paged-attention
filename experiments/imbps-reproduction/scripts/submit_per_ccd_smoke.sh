#!/usr/bin/env bash

set -euo pipefail

smoke_id="$(sbatch --parsable \
  --array=1,8%1 \
  --time=01:00:00 \
  --export="ALL,PER_CCD_ROWS=128,PER_CCD_ROUNDS=1,PER_CCD_WARMUPS=1" \
  slurm/per_ccd.sbatch)"

printf 'per_ccd_smoke=%s (global K=12 and parallel P=12/K=12; no matrix summary)\n' \
  "${smoke_id}"
