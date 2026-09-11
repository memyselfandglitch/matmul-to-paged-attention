#!/usr/bin/env bash
#SBATCH --job-name=kv-traversal
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=cpu-study-%j.out
#SBATCH --error=cpu-study-%j.err

set -euo pipefail

readonly REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS=1
export BLOCKS="${BLOCKS:-512}"
export HEADS="${HEADS:-32}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export HEAD_DIM="${HEAD_DIM:-128}"
export WARMUPS="${WARMUPS:-2}"
export REPEATS="${REPEATS:-9}"
export TRIALS="${TRIALS:-5}"
export RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/job-${SLURM_JOB_ID}}"

./scripts/run_study.sh
