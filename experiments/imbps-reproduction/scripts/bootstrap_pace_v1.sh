#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PACE_DIR="${PROJECT_ROOT}/vendor/AMD-PACE"
readonly VENV_DIR="${PROJECT_ROOT}/.venv"
readonly EXPECTED_COMMIT="cfbe8b551cca18c686b771144c18129242796ea1"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"
readonly RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

cd "${PROJECT_ROOT}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -e "${VENV_DIR}" ]]; then
  echo "Refusing to reuse existing environment: ${VENV_DIR}" >&2
  echo "Move it aside and rerun so dependency state starts clean." >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/vendor"
if [[ ! -d "${PACE_DIR}/.git" ]]; then
  git clone --branch v1.0 --depth 1 https://github.com/amd/AMD-PACE.git "${PACE_DIR}"
fi

readonly RESOLVED_COMMIT="$(git -C "${PACE_DIR}" rev-parse HEAD)"
if [[ "${RESOLVED_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "PACE checkout mismatch." >&2
  echo "Expected: ${EXPECTED_COMMIT}" >&2
  echo "Found:    ${RESOLVED_COMMIT}" >&2
  echo "Use a separate clean checkout; this script will not reset an existing tree." >&2
  exit 1
fi

if [[ -n "$(git -C "${PACE_DIR}" status --short)" ]]; then
  echo "PACE checkout has local changes; refusing to build an unrecorded variant." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${PACE_DIR}/build_requirements.txt"
"${VENV_DIR}/bin/python" -m pip install -r "${PACE_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip install --no-build-isolation -v "${PACE_DIR}"

PACE_ROOT="${PACE_DIR}" "${VENV_DIR}/bin/python" \
  src/system_info.py \
  --output "${PROJECT_ROOT}/results/post-install-${RUN_STAMP}/environment.json"

echo "PACE v1.0 installed in ${VENV_DIR}"
echo "Run: source ${VENV_DIR}/bin/activate"
