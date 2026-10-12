#!/usr/bin/env bash
#SBATCH --job-name=imbps-batched-summary
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=batched-decode-l2-summary-%j.out
#SBATCH --error=batched-decode-l2-summary-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
: "${BATCHED_DECODE_ARRAY_JOB_ID:?BATCHED_DECODE_ARRAY_JOB_ID is required}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/batched-decode-l2-${BATCHED_DECODE_ARRAY_JOB_ID}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
python3 src/summarize_batched_decode.py "${RESULT_ROOT}"
echo "Batched decode summary: ${RESULT_ROOT}/analysis"
