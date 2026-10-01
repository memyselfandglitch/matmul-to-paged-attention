#!/usr/bin/env bash
#SBATCH --job-name=imbps-build
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=bootstrap-%j.out
#SBATCH --error=bootstrap-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

srun --cpu-bind=cores ./scripts/bootstrap_pace_v1.sh
