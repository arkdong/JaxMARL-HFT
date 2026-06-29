#!/bin/bash
# Generic JaxMARL-HFT training job. Submit through snellius_scripts/submit_train.sh.
set -euo pipefail

cd /home/adong/JaxMARL-HFT

source snellius_scripts/00_cluster.sh
if [[ -z "${EXP_FILE:-}" ]]; then
  echo "ERROR: EXP_FILE is not set. Use snellius_scripts/submit_train.sh." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${EXP_FILE}"

load_snellius_python
source "${VENV_DIR}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MPLBACKEND=Agg

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax_compilation_cache"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

LOBSTER_DIR="${DATA_ROOT}/rawLOBSTER/${STOCK}/${PERIOD}"
RUN_ID="${EXP_NAME}_${SLURM_JOB_ID}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

# Reproducibility snapshot
cp "${EXP_FILE}" "${RUN_DIR}/experiment.env"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null || true
python -m pip freeze > "${RUN_DIR}/pip_freeze.txt" || true
{
  echo "RUN_ID=${RUN_ID}"
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "NODE=$(hostname)"
  echo "STARTED=$(date -Is)"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "LOBSTER_DIR=${LOBSTER_DIR}"
  echo "EXP_FILE=${EXP_FILE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
  echo "SLURM_GPUS=${SLURM_GPUS:-not set}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-not set}"
} | tee "${RUN_DIR}/manifest.txt"

if [[ ! -d "${LOBSTER_DIR}" ]]; then
  echo "ERROR: LOBSTER directory does not exist: ${LOBSTER_DIR}" >&2
  exit 2
fi
if ! find "${LOBSTER_DIR}" -maxdepth 1 -type f -name '*message_10.csv' | grep -q .; then
  echo "ERROR: no *_message_10.csv files found in ${LOBSTER_DIR}" >&2
  exit 2
fi
if ! find "${LOBSTER_DIR}" -maxdepth 1 -type f -name '*orderbook_10.csv' | grep -q .; then
  echo "ERROR: no *_orderbook_10.csv files found in ${LOBSTER_DIR}" >&2
  exit 2
fi

cat <<INFO
============================================================
JaxMARL-HFT training job
============================================================
Run ID:             ${RUN_ID}
Job ID:             ${SLURM_JOB_ID}
Node:               $(hostname)
Started:            $(date -Is)
Repo:               ${REPO_ROOT}
Data root:          ${DATA_ROOT}
LOBSTER dir:        ${LOBSTER_DIR}
Stock/period:       ${STOCK} / ${PERIOD}
Train script:       ${TRAIN_SCRIPT}
Hydra config:       ${CONFIG_NAME}
NUM_ENVS:           ${NUM_ENVS}
NUM_STEPS:          ${NUM_STEPS}
TOTAL_TIMESTEPS:    ${TOTAL_TIMESTEPS}
Expected updates:   $(( TOTAL_TIMESTEPS / NUM_STEPS / NUM_ENVS ))
N_DEVICES:          ${N_DEVICES}
WandB mode/project: ${WANDB_MODE} / ${WANDB_PROJECT}
Run dir:            ${RUN_DIR}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}
============================================================
INFO

nvidia-smi

python - <<'PY'
import os
import jax
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX local device count:", jax.local_device_count())
expected = int(os.environ.get("N_DEVICES", "0"))
assert jax.local_device_count() == expected, (
    f"Expected {expected} local GPU(s), got {jax.local_device_count()}"
)
print(f"OK: JAX can see {expected} GPU(s)")
PY

srun --cpu-bind=cores python -u "${TRAIN_SCRIPT}" \
  --config-name="${CONFIG_NAME}" \
  "++WANDB_MODE=${WANDB_MODE}" \
  "++ENTITY=${WANDB_ENTITY}" \
  "++PROJECT=${WANDB_PROJECT}" \
  "++ENV_CONFIG=${ENV_CONFIG}" \
  "++TimePeriod=${PERIOD}" \
  "++NUM_ENVS=${NUM_ENVS}" \
  "++NUM_STEPS=${NUM_STEPS}" \
  "++TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}" \
  "++N_DEVICES=${N_DEVICES}" \
  "++Timing=${TIMING}" \
  "++SEED=${SEED}"

echo "Finished: $(date -Is)" | tee -a "${RUN_DIR}/manifest.txt"
