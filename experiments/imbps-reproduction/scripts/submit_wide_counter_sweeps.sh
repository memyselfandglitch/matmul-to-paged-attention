#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
readonly DECODE_LABEL=uprof-decode-l2-split-transition
readonly CCD_LABEL=uprof-per-ccd-batch-layout
readonly DECODE_TIMING="${PWD}/results/decode-l2-split-transition-17414/timing/summary.csv"
readonly CCD_TIMING_IDS=17540+17536+17415
readonly CCD_SHAPES='12:128+13:128+12:192+13:192+12:256+13:256+12:384+13:384+12:512+13:512+8:384+16:192+16:384+24:256+23:512+24:512'

for path in "${DECODE_TIMING}" results/per-ccd-batch-layout-{17540,17536,17415}/summary.csv; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing clean timing summary: ${path}" >&2
    exit 1
  fi
done
shape_count="$(python3 src/summarize_ccd_batch_layout.py --shape-count --shapes "${CCD_SHAPES}")"
export REQUIRE_THP_MODE=always

decode_id="$(sbatch --parsable \
  --export="ALL,L2_UPROF_CLAIM=decode_l2_split_transition_opt30b,L2_UPROF_RESULT_LABEL=${DECODE_LABEL},ROUNDS=4" \
  slurm/uprof_decode_l2_core.sbatch)"
decode_summary_id="$(sbatch --parsable --dependency="afterok:${decode_id}" \
  --export="ALL,L2_UPROF_JOB_ID=${decode_id},L2_UPROF_RESULT_LABEL=${DECODE_LABEL},L2_ANALYSIS_RESULT_LABEL=${DECODE_LABEL},L2_TIMING_SUMMARY=${DECODE_TIMING}" \
  slurm/uprof_decode_l2_core_summary.sbatch)"
printf 'decode_profile=%s decode_summary=%s\n' "${decode_id}" "${decode_summary_id}"

ccd_id="$(sbatch --parsable --array="0-$((shape_count - 1))%1" \
  --export="ALL,PER_CCD_UPROF_CLAIM=opt30b_per_ccd_batch_layout,PER_CCD_UPROF_RESULT_LABEL=${CCD_LABEL},PER_CCD_LAYOUT_SHAPES=${CCD_SHAPES},PER_CCD_UPROF_REPEATS=4" \
  slurm/uprof_per_ccd_inner_threads.sbatch)"
ccd_summary_id="$(sbatch --parsable --dependency="afterok:${ccd_id}" \
  --export="ALL,PER_CCD_UPROF_ARRAY_JOB_ID=${ccd_id},PER_CCD_UPROF_RESULT_LABEL=${CCD_LABEL},PER_CCD_LAYOUT_TIMING_IDS=${CCD_TIMING_IDS}" \
  slurm/uprof_per_ccd_summary.sbatch)"
printf 'ccd_profile=%s ccd_summary=%s\n' "${ccd_id}" "${ccd_summary_id}"
