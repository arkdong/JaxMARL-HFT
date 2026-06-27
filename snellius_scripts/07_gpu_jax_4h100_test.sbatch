#!/bin/bash
#SBATCH --job-name=jmhft-4h100-test
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus=4
#SBATCH --mem=720G
#SBATCH --time=01:00:00
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

nvidia-smi

python - <<'PY'
import os
import jax

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX local device count:", jax.local_device_count())

assert jax.local_device_count() == 4, f"Expected 4 local GPUs, got {jax.local_device_count()}"
print("OK: JAX can see 4 H100 GPUs")
PY
