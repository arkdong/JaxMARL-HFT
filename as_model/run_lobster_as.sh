#!/usr/bin/env bash
set -euo pipefail

python as_model/script/run_as_baseline.py \
  --data-format lobster \
  --data-dir data/rawLOBSTER/AMZN \
  --cache-dir data/cache/as_lobster_10 \
  # --workers "${AS_WORKERS:-32}"
