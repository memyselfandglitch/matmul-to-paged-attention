#!/usr/bin/env bash

set -euo pipefail

profile_id="$(sbatch --parsable slurm/uprof_decode_l2_core.sbatch)"
summary_id="$(sbatch --parsable \
  --dependency="afterok:${profile_id}" \
  --export="ALL,L2_UPROF_JOB_ID=${profile_id}" \
  slurm/uprof_decode_l2_core_summary.sbatch)"

printf 'l2_profile=%s summary=%s\n' "${profile_id}" "${summary_id}"
