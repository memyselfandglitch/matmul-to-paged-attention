#!/usr/bin/env bash
#SBATCH --job-name=imbps-l2-own
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=l2-ownership-%j.out
#SBATCH --error=l2-ownership-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly RESULT_ROOT="${PROJECT_ROOT}/results/l2-ownership-${SLURM_JOB_ID}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
mkdir -p "${RESULT_ROOT}"

python3 src/audit_pace_imbps.py \
  --pace-root vendor/AMD-PACE \
  --output "${RESULT_ROOT}/pace-source-audit.json"

python3 src/private_l2_model.py \
  --rows 128,256,384,512,1024 \
  --splits 2,4,8,16 \
  --workers 96 \
  --l2-mib-per-worker 1 \
  --output "${RESULT_ROOT}/private-l2-bounds.json"

echo "L2 ownership analysis: ${RESULT_ROOT}"
