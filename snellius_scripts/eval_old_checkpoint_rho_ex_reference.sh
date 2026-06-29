#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/gpfs/home4/adong/JaxMARL-HFT}"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 gymnax_exchange/jaxrl/MARL/eval_checkpoint_rho_ex_reference.py \
  --config-name="eval_old_checkpoint_rho_ex_reference" \
  "$@"
