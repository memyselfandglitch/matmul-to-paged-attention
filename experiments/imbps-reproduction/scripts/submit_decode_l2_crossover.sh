#!/usr/bin/env bash

set -euo pipefail

readonly TIMING_LABEL="decode-l2-crossover"
readonly PROFILE_LABEL="uprof-decode-l2-crossover"
readonly CLAIM="decode_l2_crossover_opt30b"

timing_id="$(sbatch --parsable \
  --export="ALL,DECODE_L2_CLAIM=${CLAIM},DECODE_L2_RESULT_LABEL=${TIMING_LABEL}" \
  slurm/decode_l2.sbatch)"

profile_id="$(sbatch --parsable \
  --dependency="afterok:${timing_id}" \
  --export="ALL,L2_UPROF_CLAIM=${CLAIM},L2_UPROF_RESULT_LABEL=${PROFILE_LABEL},ROUNDS=5,WARMUPS=5,ITERATIONS=10,MIN_COUNTER_SECONDS=5" \
  slurm/uprof_decode_l2_core.sbatch)"

timing_summary="${PWD}/results/${TIMING_LABEL}-${timing_id}/timing/summary.csv"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${profile_id}" \
  --export="ALL,L2_UPROF_JOB_ID=${profile_id},L2_UPROF_RESULT_LABEL=${PROFILE_LABEL},L2_ANALYSIS_RESULT_LABEL=${PROFILE_LABEL},L2_TIMING_SUMMARY=${timing_summary}" \
  slurm/uprof_decode_l2_core_summary.sbatch)"

printf 'timing=%s l2_profile=%s summary=%s\n' \
  "${timing_id}" "${profile_id}" "${summary_id}"
