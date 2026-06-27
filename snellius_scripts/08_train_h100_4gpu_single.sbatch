#!/bin/bash
#SBATCH --job-name=jmhft-4h100-single
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=02:00:00
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
export MPLBACKEND=Agg

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

nvidia-smi

python - <<'PY'
import os
import jax

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX devices:", jax.devices())
print("JAX local device count:", jax.local_device_count())
assert jax.local_device_count() == 4, "This job should see exactly 4 GPUs"
PY

# 4-GPU weak-scaling test:
# previous 1-GPU smoke: NUM_ENVS=64
# 4 GPUs: NUM_ENVS=256 keeps 64 envs/GPU.
srun python -u \
  gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name=ippo_rnn_JAXMARL_2player \
  WANDB_MODE=disabled \
  PROJECT=h100_4gpu_single_test \
  NUM_ENVS=256 \
  TOTAL_TIMESTEPS=524288 \
  N_DEVICES=4 \
  Timing=True \
  SEED=42

echo "Finished: $(date -Is)"
