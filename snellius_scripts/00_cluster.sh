#!/usr/bin/env bash
# Stable Snellius + user environment settings.
# Keep this file. Do NOT put experiment hyperparameters here.

export REPO_ROOT="${REPO_ROOT:-/home/adong/JaxMARL-HFT}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
export CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
export LOG_DIR="${LOG_DIR:-${REPO_ROOT}/slurm_logs}"
export RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs}"

export SOFTWARE_STACK="${SOFTWARE_STACK:-AUTO}"
export PYTHON_MODULE="${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"

# Default WandB identity. Experiment files choose mode/project/name.
export WANDB_ENTITY_DEFAULT="${WANDB_ENTITY_DEFAULT:-ark-dong-university-of-amsterdam}"

echo_env_settings() {
  local vars=(
    REPO_ROOT
    DATA_ROOT
    VENV_DIR
    CACHE_ROOT
    LOG_DIR
    RUN_ROOT
    SOFTWARE_STACK
    PYTHON_MODULE
    WANDB_ENTITY_DEFAULT
  )

  local var
  for var in "${vars[@]}"; do
    echo "${var}=${!var}"
  done
}

echo_env_settings

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
      module load "${stack}" >/dev/null 2>&1 || continue
    fi
    if module load "${PYTHON_MODULE}" >/dev/null 2>&1; then
      echo "Using software stack: ${stack}"
      echo "Using Python module: ${PYTHON_MODULE}"
      python --version
      return 0
    fi
  done

  echo "ERROR: Could not load ${PYTHON_MODULE}" >&2
  echo "Try: module spider ${PYTHON_MODULE}" >&2
  return 1
}