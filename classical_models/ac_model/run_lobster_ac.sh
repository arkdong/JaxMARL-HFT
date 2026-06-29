#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

uv --project classical_models run ac-benchmark run-lobster \
  --data-format lobster \
  --data-dir data/rawLOBSTER/AMZN/lobster_amzn_10 \
  --cache-dir data/cache/ac_lobster_10 \
  --output-dir results/AC \
  --lobster-levels 10
