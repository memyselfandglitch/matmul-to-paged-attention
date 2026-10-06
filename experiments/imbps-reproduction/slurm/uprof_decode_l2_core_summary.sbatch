#!/usr/bin/env bash
#SBATCH --job-name=imbps-l2-sum
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=uprof-decode-l2-core-summary-%j.out
#SBATCH --error=uprof-decode-l2-core-summary-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
: "${L2_UPROF_JOB_ID:?L2_UPROF_JOB_ID must name the completed profiling job}"
readonly INPUT_DIR="${PROJECT_ROOT}/results/uprof-decode-l2-core-${L2_UPROF_JOB_ID}/matrix"
readonly OUTPUT_DIR="${PROJECT_ROOT}/results/uprof-decode-l2-core-${L2_UPROF_JOB_ID}-analysis-${SLURM_JOB_ID}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
python3 src/summarize_uprof_l2.py \
  --l2-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}"

echo "Decode L2 core-only summary: ${OUTPUT_DIR}"
