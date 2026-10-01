#!/usr/bin/env bash
#SBATCH --job-name=imbps-table8
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=table8-%j.out
#SBATCH --error=table8-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=passive
export LIBXSMM_BLOCK_SIZE=32
export EXPECTED_PHYSICAL_CORES="${SLURM_CPUS_PER_TASK}"
export REQUIRE_SINGLE_SOCKET=1
export REJECT_SMT=1

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  numactl --cpunodebind=0 --membind=0 ./scripts/preflight.sh
srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  numactl --cpunodebind=0 --membind=0 ./scripts/run_table_viii.sh
