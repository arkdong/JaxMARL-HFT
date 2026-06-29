#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSICAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CLASSICAL_ROOT}/.." && pwd)"

cd "${REPO_ROOT}"

uv run --project "${CLASSICAL_ROOT}" python "${SCRIPT_DIR}/script/run_as_baseline.py" \
  --data-format lobster \
  --data-dir "${REPO_ROOT}/data/rawLOBSTER/AMZN/lobster_amzn_10" \
  --cache-dir "${REPO_ROOT}/data/cache/as_lobster_10" \
  --as-horizon-mode finite_episode
