#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_config.sh"

mkdir -p \
  "${DATA_ROOT}/rawLOBSTER/${STOCK}/${PERIOD}" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/jax" \
  "${LOG_DIR}"

cat <<EOF
Prepared directories:
  REPO_ROOT = ${REPO_ROOT}
  DATA_ROOT = ${DATA_ROOT}
  LOBSTER   = ${DATA_ROOT}/rawLOBSTER/${STOCK}/${PERIOD}
  VENV_DIR  = ${VENV_DIR}
  LOG_DIR   = ${LOG_DIR}

Current quota summary, if myquota is available:
EOF

if command -v myquota >/dev/null 2>&1; then
  myquota || true
else
  echo "myquota not found in PATH; this is okay, but check quota manually on Snellius."
fi

echo
echo "Current repo/data sizes:"
du -sh "${REPO_ROOT}" 2>/dev/null || true
du -sh "${DATA_ROOT}" 2>/dev/null || true
