#!/usr/bin/env bash
set -euo pipefail

cd /home/adong/JaxMARL-HFT
mkdir -p snellius_scripts slurm_logs

stamp="$(date +%Y%m%d_%H%M%S)"
for f in \
  snellius_scripts/00_config.sh \
  snellius_scripts/04_gpu_jax_test.sbatch \
  snellius_scripts/05_smoke_train.sbatch \
  snellius_scripts/06_train_h100.sbatch; do
  if [[ -f "$f" ]]; then
    cp "$f" "$f.bak.${stamp}"
  fi
done

cat > snellius_scripts/00_config.sh <<'EOF_CONFIG_H100'
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
export PERIOD="${PERIOD:-2024}"

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
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
export TRAIN_PROJECT="${TRAIN_PROJECT:-amzn_2024_ippo_h100}"

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
EOF_CONFIG_H100

cat > snellius_scripts/04_gpu_jax_test.sbatch <<'EOF_GPU_TEST_H100'
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
EOF_GPU_TEST_H100

cat > snellius_scripts/05_smoke_train.sbatch <<'EOF_SMOKE_H100'
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
EOF_SMOKE_H100

cat > snellius_scripts/06_train_h100.sbatch <<'EOF_TRAIN_H100'
#!/bin/bash
#SBATCH --job-name=jmhft-h100-train
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
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
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
echo "NUM_ENVS: ${TRAIN_NUM_ENVS}"
echo "TOTAL_TIMESTEPS: ${TRAIN_TOTAL_TIMESTEPS}"
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
  ENTITY="${WANDB_ENTITY}" \
  PROJECT="${TRAIN_PROJECT}" \
  N_DEVICES=1 \
  NUM_ENVS="${TRAIN_NUM_ENVS}" \
  TOTAL_TIMESTEPS="${TRAIN_TOTAL_TIMESTEPS}" \
  SEED="${TRAIN_SEED}"

echo "Finished: $(date -Is)"
EOF_TRAIN_H100

chmod +x snellius_scripts/00_config.sh
chmod +x snellius_scripts/04_gpu_jax_test.sbatch
chmod +x snellius_scripts/05_smoke_train.sbatch
chmod +x snellius_scripts/06_train_h100.sbatch

cat <<'EOF_DONE'
H100 single-node scripts installed.

Changed/created:
  snellius_scripts/00_config.sh
  snellius_scripts/04_gpu_jax_test.sbatch
  snellius_scripts/05_smoke_train.sbatch
  snellius_scripts/06_train_h100.sbatch

Next commands:
  source snellius_scripts/00_config.sh
  load_snellius_python
  bash snellius_scripts/02_setup_python_env.sh      # only if .venv is not already ready
  bash snellius_scripts/03_configure_jaxmarl_hft.sh # only if not already patched for paths
  sbatch snellius_scripts/04_gpu_jax_test.sbatch
  sbatch snellius_scripts/05_smoke_train.sbatch
  sbatch snellius_scripts/06_train_h100.sbatch
EOF_DONE
