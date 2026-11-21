#!/usr/bin/env bash

set -euo pipefail

case "${1:-}" in
  decode)
    sbatch --parsable \
      --export=ALL,DECODE_L2_CLAIM=decode_l2_split_transition_opt30b,DECODE_L2_RESULT_LABEL=decode-l2-split-transition \
      slurm/decode_l2.sbatch
    ;;
  ccd)
    shapes="${2:-}"
    count="$(python3 src/summarize_ccd_batch_layout.py --shape-count --shapes "${shapes}")"
    default_pairs=3
    if [[ -n "${shapes}" ]]; then
      default_pairs=9
    fi
    pairs="${PER_CCD_LAYOUT_PAIRS:-${default_pairs}}"
    if [[ ! "${pairs}" =~ ^[1-9][0-9]*$ ]] || ((pairs < 3)); then
      echo "PER_CCD_LAYOUT_PAIRS must be an integer of at least 3" >&2
      exit 2
    fi
    settings="PER_CCD_LAYOUT_PAIRS=${pairs},PER_CCD_LAYOUT_SHAPES=${shapes}"
    matrix_id="$(sbatch --parsable \
      --array="0-$((count - 1))%1" \
      --export="ALL,PER_CCD_CLAIM=opt30b_per_ccd_batch_layout,${settings}" \
      slurm/per_ccd_inner_threads.sbatch)"
    summary_id="$(sbatch --parsable --dependency="afterok:${matrix_id}" \
      --export="ALL,PER_CCD_ARRAY_JOB_ID=${matrix_id},PER_CCD_RESULT_LABEL=per-ccd-batch-layout,${settings}" \
      slurm/per_ccd_summary.sbatch)"
    printf 'ccd=%s summary=%s\n' "${matrix_id}" "${summary_id}"
    ;;
  *)
    echo "Usage: bash scripts/submit_split_followups.sh decode | ccd [B:Q+B:Q]" >&2
    exit 2
    ;;
esac
