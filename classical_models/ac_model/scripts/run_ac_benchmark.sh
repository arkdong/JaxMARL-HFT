#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end commands. Edit the paths and levels to match your files.
# If your snapshot files are already sampled once per RL step, set MSGS_PER_STEP=1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSICAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CLASSICAL_ROOT}"

LEVELS=${LEVELS:-10}
MSGS_PER_STEP=${MSGS_PER_STEP:-100}
TASK_SIZE=${TASK_SIZE:-600}
EPISODE_LENGTH=${EPISODE_LENGTH:-64}
LOT_SIZE=${LOT_SIZE:-10}
TICK_SIZE=${TICK_SIZE:-0.01}

TRAIN=${TRAIN:-data/amzn_train.parquet}
VALID=${VALID:-data/amzn_valid.parquet}
TEST=${TEST:-data/amzn_test.parquet}

uv run ac-benchmark calibrate \
  --snapshots "$TRAIN" \
  --out-dir outputs/ac/train_calibration \
  --levels "$LEVELS" \
  --messages-per-step "$MSGS_PER_STEP" \
  --q-grid 10,20,50,100,200,400,600

uv run ac-benchmark evaluate \
  --snapshots "$VALID" \
  --out-dir outputs/ac/validation_sweep \
  --levels "$LEVELS" \
  --task-size "$TASK_SIZE" \
  --episode-length "$EPISODE_LENGTH" \
  --messages-per-step "$MSGS_PER_STEP" \
  --episode-start-frequency-steps 64 \
  --lot-size "$LOT_SIZE" \
  --tick-size "$TICK_SIZE" \
  --directions random \
  --kappa-grid 0,0.5,1,2,4

uv run ac-benchmark evaluate \
  --snapshots "$TEST" \
  --out-dir outputs/ac/test_benchmark \
  --levels "$LEVELS" \
  --task-size "$TASK_SIZE" \
  --episode-length "$EPISODE_LENGTH" \
  --messages-per-step "$MSGS_PER_STEP" \
  --episode-start-frequency-steps 64 \
  --lot-size "$LOT_SIZE" \
  --tick-size "$TICK_SIZE" \
  --directions random \
  --kappa-grid 0,1,2
