#!/usr/bin/env bash
#SBATCH --job-name=imbps-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --output=preflight-%j.out
#SBATCH --error=preflight-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
export EXPECTED_PHYSICAL_CORES="${SLURM_CPUS_PER_TASK}"
export REQUIRE_SINGLE_SOCKET=1
export REJECT_SMT=1
srun --cpu-bind=cores ./scripts/preflight.sh
