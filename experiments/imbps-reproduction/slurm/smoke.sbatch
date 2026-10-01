#!/usr/bin/env bash
#SBATCH --job-name=imbps-smoke
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=smoke-%j.out
#SBATCH --error=smoke-%j.err

set -euo pipefail

readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
readonly RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RESULT_DIR="${PROJECT_ROOT}/results/smoke-${RUN_STAMP}"
cd "${PROJECT_ROOT}"
source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=passive
export LIBXSMM_BLOCK_SIZE=32

mkdir -p "${RESULT_DIR}"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  numactl --cpunodebind=0 --membind=0 python src/cache_model.py \
  --batch 16 --sequence 1920 --hidden 7168 --intermediate 28672 \
  --bytes-per-element 2 --cache-mib 512 --splits 4 \
  --json > "${RESULT_DIR}/cache-512.json"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  numactl --cpunodebind=0 --membind=0 python src/cache_model.py \
  --batch 16 --sequence 1920 --hidden 7168 --intermediate 28672 \
  --bytes-per-element 2 --cache-mib 750 --splits 4 \
  --json > "${RESULT_DIR}/cache-750.json"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  numactl --cpunodebind=0 --membind=0 python src/pace_mlp_bench.py \
  --backend tpp --batch 1 --sequence 64 \
  --hidden 768 --intermediate 3072 --splits 1 --activation relu \
  --warmups 2 --iterations 5 --output "${RESULT_DIR}/opt125m-tpp.json"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  numactl --cpunodebind=0 --membind=0 python src/pace_mlp_bench.py \
  --backend imbps --batch 1 --sequence 64 \
  --hidden 768 --intermediate 3072 --splits 4 --activation relu \
  --warmups 2 --iterations 5 --output "${RESULT_DIR}/opt125m-imbps-k4.json"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  numactl --cpunodebind=0 --membind=0 python src/check_numerics.py \
  --hidden 768 --intermediate 3072 --rows 128 \
  --splits 1,2,4,8,16 --dtype bf16 --activation relu \
  --output "${RESULT_DIR}/opt125m-numerics.json"

echo "Smoke results: ${RESULT_DIR}"
