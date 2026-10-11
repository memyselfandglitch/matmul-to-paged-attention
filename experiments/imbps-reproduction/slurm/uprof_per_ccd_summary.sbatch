#!/usr/bin/env bash
#SBATCH --job-name=imbps-ccd-l3-sum
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=uprof-per-ccd-summary-%j.out
#SBATCH --error=uprof-per-ccd-summary-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
: "${PER_CCD_UPROF_ARRAY_JOB_ID:?PER_CCD_UPROF_ARRAY_JOB_ID is required}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/uprof-per-ccd-threads-${PER_CCD_UPROF_ARRAY_JOB_ID}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
python3 src/summarize_uprof_per_ccd.py "${RESULT_ROOT}"
echo "CCD-level uProf summary: ${RESULT_ROOT}/analysis"
