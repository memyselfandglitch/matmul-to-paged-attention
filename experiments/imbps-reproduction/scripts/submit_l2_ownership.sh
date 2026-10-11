#!/usr/bin/env bash

set -euo pipefail

job_id="$(sbatch --parsable slurm/l2_ownership.sbatch)"
printf 'l2_ownership=%s\n' "${job_id}"
