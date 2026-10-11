#!/usr/bin/env bash

set -euo pipefail

smoke_id="$(sbatch --parsable \
  --array=0,4%1 \
  --time=01:00:00 \
  --export="ALL,FULL_DECODE_CONTEXTS=16,FULL_DECODE_OUTPUT_TOKENS=4,FULL_DECODE_ROUNDS=1,FULL_DECODE_WARMUPS=0" \
  slurm/full_decode.sbatch)"

printf 'full_decode_smoke=%s (normal TPP and PARD-4 TPP; no matrix summary)\n' \
  "${smoke_id}"
