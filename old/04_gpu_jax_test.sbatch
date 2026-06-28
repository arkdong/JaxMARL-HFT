#!/bin/bash
#SBATCH --job-name=jmhft-h100-gpu-test
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/home/adong/JaxMARL-HFT/slurm_logs/%x-%j.out
#SBATCH --error=/home/adong/JaxMARL-HFT/slurm_logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="/home/adong/JaxMARL-HFT/snellius_scripts"
source "${SCRIPT_DIR}/00_config.sh"
load_snellius_python
source "${VENV_DIR}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date -Is)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "GPUs requested: ${SLURM_GPUS:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"

nvidia-smi

python - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
assert any(d.platform in {"gpu", "cuda"} for d in jax.devices()), "No GPU visible to JAX"
print("OK: JAX can see a GPU")
PY
