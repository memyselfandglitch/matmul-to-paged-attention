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
readonly PROFILE_LABEL="${L2_UPROF_RESULT_LABEL:-uprof-decode-l2-core}"
readonly ANALYSIS_LABEL="${L2_ANALYSIS_RESULT_LABEL:-uprof-decode-l2-core}"
readonly INPUT_DIR="${PROJECT_ROOT}/results/${PROFILE_LABEL}-${L2_UPROF_JOB_ID}/matrix"
readonly OUTPUT_DIR="${PROJECT_ROOT}/results/${ANALYSIS_LABEL}-${L2_UPROF_JOB_ID}-analysis-${SLURM_JOB_ID}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
summary_args=(
  --l2-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${L2_TIMING_SUMMARY:-}" ]]; then
  summary_args+=(--timing-summary "${L2_TIMING_SUMMARY}")
fi
python3 src/summarize_uprof_l2.py "${summary_args[@]}"

echo "Decode L2 core-only summary: ${OUTPUT_DIR}"
