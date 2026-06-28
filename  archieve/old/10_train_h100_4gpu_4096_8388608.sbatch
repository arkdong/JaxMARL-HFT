#!/bin/bash
#SBATCH --job-name=jmhft-4h100-4096
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=04:00:00
#SBATCH --output=/home/adong/JaxMARL-HFT/slurm_logs/%x-%j.out
#SBATCH --error=/home/adong/JaxMARL-HFT/slurm_logs/%x-%j.err

set -euo pipefail

cd /home/adong/JaxMARL-HFT

source snellius_scripts/00_config.sh
load_snellius_python
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MPLBACKEND=Agg

# Keep CPU libraries from oversubscribing the 64 CPU cores on the H100 node.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# These are harmless if your Python source already sets them; if you patched
# ippo_rnn_JAXMARL.py to use os.environ.setdefault(...), these values apply.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

NUM_ENVS=4096
TOTAL_TIMESTEPS=8388608
NUM_STEPS=64
N_DEVICES=4
SEED=42
PROJECT_NAME="h100_4gpu_4096_8388608_test"

LOBSTER_DIR="$DATA_ROOT/rawLOBSTER/$STOCK/$PERIOD"

mkdir -p "$LOG_DIR"

cat <<INFO
============================================================
JaxMARL-HFT 4x H100 training job
============================================================
Job ID:             ${SLURM_JOB_ID}
Node:               $(hostname)
Started:            $(date -Is)
Repo:               $REPO_ROOT
Data root:          $DATA_ROOT
LOBSTER dir:        $LOBSTER_DIR
Stock/period:       $STOCK / $PERIOD
NUM_ENVS:           $NUM_ENVS
NUM_STEPS:          $NUM_STEPS
TOTAL_TIMESTEPS:    $TOTAL_TIMESTEPS
Expected updates:   $(( TOTAL_TIMESTEPS / NUM_STEPS / NUM_ENVS ))
Requested GPUs:     ${SLURM_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}
============================================================
INFO

if [[ ! -d "$LOBSTER_DIR" ]]; then
  echo "ERROR: LOBSTER directory does not exist: $LOBSTER_DIR" >&2
  exit 2
fi

if ! find "$LOBSTER_DIR" -maxdepth 1 -type f -name '*message_10.csv' | grep -q .; then
  echo "ERROR: no *_message_10.csv files found in $LOBSTER_DIR" >&2
  exit 2
fi

if ! find "$LOBSTER_DIR" -maxdepth 1 -type f -name '*orderbook_10.csv' | grep -q .; then
  echo "ERROR: no *_orderbook_10.csv files found in $LOBSTER_DIR" >&2
  exit 2
fi

echo "GPU status before training:"
nvidia-smi

python - <<'PY'
import os
import jax

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX local device count:", jax.local_device_count())

assert jax.local_device_count() == 4, (
    f"Expected exactly 4 local H100 GPUs, got {jax.local_device_count()}"
)
print("OK: JAX can see 4 GPUs")
PY

# This uses the normal JAX sharded training path. With 4 visible GPUs, the code
# should print: "Multi-device: 4 device(s) detected, sharding 4096 envs across them".
srun --cpu-bind=cores python -u \
  gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name=ippo_rnn_JAXMARL_2player \
  WANDB_MODE=online \
  PROJECT="$PROJECT_NAME" \
  NUM_ENVS="$NUM_ENVS" \
  NUM_STEPS="$NUM_STEPS" \
  TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
  N_DEVICES="$N_DEVICES" \
  Timing=True \
  SEED="$SEED"

echo "Finished: $(date -Is)"
