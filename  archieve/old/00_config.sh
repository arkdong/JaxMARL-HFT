#!/usr/bin/env bash
# Shared configuration for running JaxMARL-HFT on Snellius: single-node, one-H100 setup.

export REPO_ROOT="${REPO_ROOT:-/home/adong/JaxMARL-HFT}"

# Keep data under the repo by default. For large data, later change to e.g.:
#   export DATA_ROOT="/scratch-shared/adong/jaxmarl-hft-data"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"

export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
export CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/slurm_logs}"

# Change these if your LOBSTER folder is not data/rawLOBSTER/AMZN/2024.
export STOCK="${STOCK:-AMZN}"
export PERIOD="${PERIOD:-train}"

# Your module spider output confirmed this module exists.
export SOFTWARE_STACK="${SOFTWARE_STACK:-AUTO}"
export PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"

# Single-node H100 defaults.
export GPU_PARTITION="${GPU_PARTITION:-gpu_h100}"
export GPU_COUNT="${GPU_COUNT:-1}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
export MEM_PER_JOB="${MEM_PER_JOB:-180G}"

# Training defaults. Start small; increase after smoke test succeeds.
export SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-64}"
export SMOKE_TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-50000}"
export TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-1024}"
export TRAIN_TOTAL_TIMESTEPS="${TRAIN_TOTAL_TIMESTEPS:-500000000}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-ark-dong-university-of-amsterdam}"
export TRAIN_PROJECT="${TRAIN_PROJECT:-Thesis}"

load_snellius_python() {
  local stacks=()

  if [[ "${SOFTWARE_STACK}" == "AUTO" || -z "${SOFTWARE_STACK}" ]]; then
    stacks=(2025 2024 2023 2022 none)
  else
    stacks=("${SOFTWARE_STACK}")
  fi

  local stack
  for stack in "${stacks[@]}"; do
    module purge

    if [[ "${stack}" != "none" ]]; then
      if ! module load "${stack}" >/dev/null 2>&1; then
        continue
      fi
    fi

    if module load "${PYTHON_MODULE}" >/dev/null 2>&1; then
      echo "Using software stack: ${stack}"
      echo "Using Python module: ${PYTHON_MODULE}"
      python --version
      return 0
    fi
  done

  echo "ERROR: Could not load ${PYTHON_MODULE}" >&2
  echo "Run this for details:" >&2
  echo "  module spider ${PYTHON_MODULE}" >&2
  echo "Then set SOFTWARE_STACK in snellius_scripts/00_config.sh if needed." >&2
  return 1
}
