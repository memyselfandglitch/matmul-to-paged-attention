#!/usr/bin/env bash
#SBATCH --job-name=imbps-target-summary
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=target-verify-summary-%j.out
#SBATCH --error=target-verify-summary-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
: "${TARGET_VERIFY_ARRAY_JOB_ID:?TARGET_VERIFY_ARRAY_JOB_ID must name the completed array}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/target-verify-${TARGET_VERIFY_ARRAY_JOB_ID}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
python3 src/summarize_target_verify.py "${RESULT_ROOT}"
echo "Target verification summary: ${RESULT_ROOT}/analysis"
