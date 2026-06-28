#!/bin/bash
#SBATCH --job-name=jmhft-h100-smoke
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
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.90"
export XLA_PYTHON_CLIENT_PREALLOCATE="true"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date -Is)"
echo "Repo: ${REPO_ROOT}"
echo "Data: ${DATA_ROOT}/rawLOBSTER/${STOCK}/${PERIOD}"
echo "NUM_ENVS: ${SMOKE_NUM_ENVS}"
echo "TOTAL_TIMESTEPS: ${SMOKE_TOTAL_TIMESTEPS}"
git rev-parse HEAD || true
nvidia-smi

python - <<'PY'
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
assert any(d.platform in {"gpu", "cuda"} for d in jax.devices()), "No GPU visible to JAX"
PY

srun python -u gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
  --config-name=ippo_rnn_JAXMARL_2player \
  WANDB_MODE="${WANDB_MODE}" \
  PROJECT="smoke_${STOCK}_${PERIOD}_h100" \
  N_DEVICES=1 \
  NUM_ENVS="${SMOKE_NUM_ENVS}" \
  TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS}" \
  SEED="${TRAIN_SEED}"

echo "Finished: $(date -Is)"
