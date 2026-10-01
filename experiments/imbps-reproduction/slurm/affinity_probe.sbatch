#!/usr/bin/env bash
#SBATCH --job-name=imbps-affinity
#SBATCH --partition=jobmn01
#SBATCH --nodelist=mn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --exclusive
#SBATCH --output=affinity-probe-%j.out
#SBATCH --error=affinity-probe-%j.err

set -uo pipefail

readonly AFFINITY_CODE='import os; a=sorted(os.sched_getaffinity(0)); print({"count": len(a), "cpus": a})'

echo "=== Slurm configuration ==="
scontrol show config | grep -E '^(SelectType|SelectTypeParameters|TaskPlugin|TaskPluginParam)' || true

echo "=== Relevant environment ==="
env | grep -E '^(SLURM_(CPUS|CPU|JOB|NTASKS|TASKS)|OMP_)' | sort || true

echo "=== Batch shell ==="
python3 -c "${AFFINITY_CODE}"

echo "=== Direct numactl ==="
numactl --cpunodebind=0 --membind=0 python3 -c "${AFFINITY_CODE}"
