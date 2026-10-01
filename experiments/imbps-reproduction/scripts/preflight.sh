#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RESULT_DIR="${PROJECT_ROOT}/results/preflight-${RUN_STAMP}"

mkdir -p "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

python3 src/system_info.py --output "${RESULT_DIR}/environment.json"

topology_args=(--output "${RESULT_DIR}/topology-validation.json")
if [[ -n "${EXPECTED_PHYSICAL_CORES:-}" ]]; then
  topology_args+=(--expected-physical-cores "${EXPECTED_PHYSICAL_CORES}")
fi
if [[ "${REQUIRE_SINGLE_SOCKET:-0}" == "1" ]]; then
  topology_args+=(--require-single-socket)
fi
if [[ "${REJECT_SMT:-0}" == "1" ]]; then
  topology_args+=(--reject-smt)
fi
python3 src/validate_topology.py "${topology_args[@]}"

if command -v perf >/dev/null 2>&1; then
  perf list > "${RESULT_DIR}/perf-list.txt"
fi

if command -v env >/dev/null 2>&1; then
  env | grep -E '^(OMP_|GOMP_|KMP_|IMBPS_|LIBXSMM_|DNNL_|MALLOC_CONF|SLURM_)' \
    | sort > "${RESULT_DIR}/benchmark-environment.txt" || true
fi

echo "Preflight: ${RESULT_DIR}"
