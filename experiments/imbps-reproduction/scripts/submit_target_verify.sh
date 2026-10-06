#!/usr/bin/env bash

set -euo pipefail

readonly CLAIM="${TARGET_VERIFY_CLAIM:-target_verify_opt30b_pilot}"
sbatch_overrides=()
if [[ "${CLAIM}" == "target_verify_smoke_opt125m" ]]; then
  sbatch_overrides+=(--mem=32G --time=00:30:00)
fi
timing_id="$(sbatch --parsable \
  "${sbatch_overrides[@]}" \
  --export="ALL,TARGET_VERIFY_CLAIM=${CLAIM}" \
  slurm/target_verify.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${timing_id}" \
  --export="ALL,TARGET_VERIFY_ARRAY_JOB_ID=${timing_id}" \
  slurm/target_verify_summary.sbatch)"

echo "target_verify=${timing_id} summary=${summary_id} claim=${CLAIM}"
