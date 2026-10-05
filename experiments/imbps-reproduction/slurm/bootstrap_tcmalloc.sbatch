#!/usr/bin/env bash
#SBATCH --job-name=imbps-tcmalloc
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=tcmalloc-%j.out
#SBATCH --error=tcmalloc-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

./scripts/bootstrap_tcmalloc.sh
