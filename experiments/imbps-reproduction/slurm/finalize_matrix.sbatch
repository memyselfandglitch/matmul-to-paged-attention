#!/usr/bin/env bash
#SBATCH --job-name=imbps-finalize
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=finalize-%j.out
#SBATCH --error=finalize-%j.err

set -euo pipefail

: "${CLAIM:?submit with CLAIM=table_ii or CLAIM=table_viii}"
: "${RESULT_DIR:?submit with RESULT_DIR pointing to the existing result directory}"

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

python3 src/run_standalone_matrix.py \
  --claim "${CLAIM}" \
  --finalize-existing "${RESULT_DIR}"
