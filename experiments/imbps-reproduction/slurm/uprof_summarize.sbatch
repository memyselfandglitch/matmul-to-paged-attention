#!/usr/bin/env bash
#SBATCH --job-name=imbps-upr-sum
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=uprof-summary-%j.out
#SBATCH --error=uprof-summary-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

if [[ -z "${UPROF_ARRAY_JOB_ID:-}" ]]; then
  echo "UPROF_ARRAY_JOB_ID is required" >&2
  exit 2
fi

readonly RESULT_LABEL="${UPROF_RESULT_LABEL:-table-ii}"
readonly CACHE_DIR="${PROJECT_ROOT}/results/uprof-${RESULT_LABEL}-${UPROF_ARRAY_JOB_ID}-cache/matrix"
readonly TRAFFIC_DIR="${PROJECT_ROOT}/results/uprof-${RESULT_LABEL}-${UPROF_ARRAY_JOB_ID}-traffic/matrix"
readonly OUTPUT_DIR="${PROJECT_ROOT}/results/uprof-${RESULT_LABEL}-${UPROF_ARRAY_JOB_ID}-analysis-${SLURM_JOB_ID}"

python3 src/summarize_uprof_matrix.py \
  --cache-dir "${CACHE_DIR}" \
  --traffic-dir "${TRAFFIC_DIR}" \
  --output-dir "${OUTPUT_DIR}"
