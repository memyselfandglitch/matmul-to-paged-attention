#!/usr/bin/env bash

set -euo pipefail

readonly TIMING_LABEL="decode-l3-sweep"
readonly PROFILE_LABEL="decode-l3-sweep"
readonly CLAIM="decode_l3_sweep_opt30b"

timing_id="$(sbatch --parsable \
  --export="ALL,DECODE_L2_CLAIM=${CLAIM},DECODE_L2_RESULT_LABEL=${TIMING_LABEL},EXPECTED_L3_MIB=384" \
  slurm/decode_l2.sbatch)"

profile_id="$(sbatch --parsable \
  --dependency="afterok:${timing_id}" \
  --export="ALL,DECODE_UPROF_CLAIM=${CLAIM},DECODE_UPROF_RESULT_LABEL=${PROFILE_LABEL},EXPECTED_L3_MIB=384,ROUNDS=5,WARMUPS=5,ITERATIONS=10,MIN_COUNTER_SECONDS=5" \
  slurm/uprof_decode_l2.sbatch)"

timing_summary="${PWD}/results/${TIMING_LABEL}-${timing_id}/timing/summary.csv"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${profile_id}" \
  --export="ALL,UPROF_ARRAY_JOB_ID=${profile_id},UPROF_RESULT_LABEL=${PROFILE_LABEL},UPROF_TIMING_SUMMARY=${timing_summary}" \
  slurm/uprof_summarize.sbatch)"

printf 'timing=%s cache_traffic_profile=%s summary=%s\n' \
  "${timing_id}" "${profile_id}" "${summary_id}"
