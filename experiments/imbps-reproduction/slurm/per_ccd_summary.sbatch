#!/usr/bin/env bash
#SBATCH --job-name=imbps-ccd-summary
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=per-ccd-summary-%j.out
#SBATCH --error=per-ccd-summary-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
: "${PER_CCD_ARRAY_JOB_ID:?PER_CCD_ARRAY_JOB_ID is required}"
readonly RESULT_LABEL="${PER_CCD_RESULT_LABEL:-per-ccd}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/${RESULT_LABEL}-${PER_CCD_ARRAY_JOB_ID}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
if [[ "${RESULT_LABEL}" == per-ccd-batch-layout ]]; then
  python3 src/summarize_ccd_batch_layout.py \
    --result-root "${RESULT_ROOT}" --pairs "${PER_CCD_LAYOUT_PAIRS:-3}" \
    --shapes "${PER_CCD_LAYOUT_SHAPES:-}"
  exit
fi
python3 src/summarize_per_ccd.py "${RESULT_ROOT}"
echo "Per-CCD summary: ${RESULT_ROOT}/analysis"
