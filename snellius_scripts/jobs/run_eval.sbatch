#!/bin/bash
# Generic JaxMARL-HFT validation job. Submit through snellius_scripts/submit_eval.sh.
set -euo pipefail

cd /home/adong/JaxMARL-HFT

source snellius_scripts/00_cluster.sh
if [[ -z "${EXP_FILE:-}" ]]; then
  echo "ERROR: EXP_FILE is not set. Use snellius_scripts/submit_eval.sh." >&2
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

if [[ ! -f "${ENV_CONFIG}" ]]; then
  echo "ERROR: environment config does not exist: ${ENV_CONFIG}" >&2
  exit 2
fi

CONFIG_DATA_PATH="$(
  python -c 'import json, os; cfg=json.load(open(os.environ["ENV_CONFIG"])); print(cfg["world_config"].get("dataPath", os.environ["DATA_ROOT"]))'
)"
CONFIG_ALPHATRADE_PATH="$(
  python -c 'import json, os; cfg=json.load(open(os.environ["ENV_CONFIG"])); print(cfg["world_config"].get("alphatradePath", os.environ["REPO_ROOT"]))'
)"

LOBSTER_DIR="${CONFIG_DATA_PATH}/rawLOBSTER/${STOCK}/${PERIOD}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${CONFIG_ALPHATRADE_PATH}/checkpoints/MARLCheckpoints}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RESTORE_PROJECT}/${RESTORE_RUN}"
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
  echo "CONFIG_DATA_PATH=${CONFIG_DATA_PATH}"
  echo "CONFIG_ALPHATRADE_PATH=${CONFIG_ALPHATRADE_PATH}"
  echo "LOBSTER_DIR=${LOBSTER_DIR}"
  echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
  echo "RESTORE_STEP=${RESTORE_STEP:-latest}"
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
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 2
fi
if [[ -n "${RESTORE_STEP:-}" && "${RESTORE_STEP}" != "latest" && "${RESTORE_STEP}" != "none" ]]; then
  if [[ ! -d "${CHECKPOINT_DIR}/${RESTORE_STEP}" ]]; then
    echo "ERROR: checkpoint step does not exist: ${CHECKPOINT_DIR}/${RESTORE_STEP}" >&2
    exit 2
  fi
fi

cat <<INFO
============================================================
JaxMARL-HFT validation job
============================================================
Run ID:             ${RUN_ID}
Job ID:             ${SLURM_JOB_ID}
Node:               $(hostname)
Started:            $(date -Is)
Repo:               ${REPO_ROOT}
Data root:          ${CONFIG_DATA_PATH}
LOBSTER dir:        ${LOBSTER_DIR}
Stock/period:       ${STOCK} / ${PERIOD}
Eval script:        ${EVAL_SCRIPT}
Hydra config:       ${CONFIG_NAME}
NUM_ENVS:           ${NUM_ENVS}
NUM_STEPS:          ${NUM_STEPS}
N_DEVICES:          ${N_DEVICES}
Restore:            ${RESTORE_PROJECT}/${RESTORE_RUN}/${RESTORE_STEP:-latest}
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

srun --cpu-bind=cores python -u "${EVAL_SCRIPT}" \
  --config-name="${CONFIG_NAME}" \
  hydra.run.dir=. \
  ENV_CONFIG="${ENV_CONFIG}" \
  EvalTimePeriod="${PERIOD}" \
  RESTORE_PROJECT="${RESTORE_PROJECT}" \
  RESTORE_RUN="${RESTORE_RUN}" \
  RESTORE_STEP="${RESTORE_STEP:-latest}" \
  NUM_ENVS="${NUM_ENVS}" \
  NUM_STEPS="${NUM_STEPS}" \
  NUM_UPDATES="${NUM_UPDATES}" \
  N_DEVICES="${N_DEVICES}" \
  SEED="${SEED}" \
  GRU_HIDDEN_DIM="${GRU_HIDDEN_DIM}" \
  FC_DIM_SIZE="${FC_DIM_SIZE}" \
  WANDB_MODE="${WANDB_MODE}" \
  ENTITY="${WANDB_ENTITY}" \
  PROJECT="${WANDB_PROJECT}"

echo "Finished: $(date -Is)" | tee -a "${RUN_DIR}/manifest.txt"
