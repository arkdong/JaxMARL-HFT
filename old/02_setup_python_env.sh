#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: REPO_ROOT does not exist: ${REPO_ROOT}" >&2
  echo "Clone or move JaxMARL-HFT there first." >&2
  exit 1
fi

load_snellius_python

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
mkdir -p "${PIP_CACHE_DIR}"

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

cd "${REPO_ROOT}"
python -m pip install -r requirements.txt
python -m pip check || true
python -m pip freeze > "${REPO_ROOT}/requirements-lock.txt"

echo "Python environment ready: ${VENV_DIR}"
echo "Lock file written: ${REPO_ROOT}/requirements-lock.txt"
