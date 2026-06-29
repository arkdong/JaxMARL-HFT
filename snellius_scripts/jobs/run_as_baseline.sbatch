#!/bin/bash
# Generic AS classical-control baseline job.
# Submit through snellius_scripts/submit_as_baseline.sh, or run interactively with:
#   srun <resources> --export=ALL,EXP_FILE=snellius_scripts/experiments/as_lobster_baseline.env \
#     bash snellius_scripts/jobs/run_as_baseline.sbatch
set -euo pipefail

cd /home/adong/JaxMARL-HFT

source snellius_scripts/00_cluster.sh
if [[ -z "${EXP_FILE:-}" ]]; then
  echo "ERROR: EXP_FILE is not set. Use snellius_scripts/submit_as_baseline.sh or pass --export=ALL,EXP_FILE=..." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${EXP_FILE}"

load_snellius_python
source "${VENV_DIR}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/as_model:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${CACHE_ROOT}" "${RUN_ROOT}"

AS_DATA_FORMAT="${AS_DATA_FORMAT:-lobster}"
AS_DATA_DIR="${AS_DATA_DIR:-${DATA_ROOT}/rawLOBSTER/${STOCK:-AMZN}}"
AS_CACHE_DIR="${AS_CACHE_DIR:-${CACHE_ROOT}/as_lobster_10}"
AS_HORIZON_MODE="${AS_HORIZON_MODE:-finite_episode}"
AS_WORKERS="${AS_WORKERS:-1}"
AS_LOBSTER_LEVELS="${AS_LOBSTER_LEVELS:-10}"
AS_ORDER_SIZE="${AS_ORDER_SIZE:-10}"
AS_INVENTORY_LIMIT="${AS_INVENTORY_LIMIT:-200}"
AS_DECISION_INTERVAL_MS="${AS_DECISION_INTERVAL_MS:-250}"
AS_EPISODE_STEPS="${AS_EPISODE_STEPS:-64}"
AS_TICK_SIZE="${AS_TICK_SIZE:-0.01}"
AS_Q_REF="${AS_Q_REF:-20.0}"
AS_TARGET_SKEW="${AS_TARGET_SKEW:-0.05}"
AS_DELTA_START_TICKS="${AS_DELTA_START_TICKS:-1}"
AS_DELTA_END_TICKS="${AS_DELTA_END_TICKS:-20}"

RUN_ID="${EXP_NAME}_${SLURM_JOB_ID:-manual_$(date +%Y%m%dT%H%M%S)}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
AS_OUTPUT_DIR="${AS_OUTPUT_DIR:-}"
AS_BASELINES_OUTPUT_DIR="${AS_BASELINES_OUTPUT_DIR:-}"
if [[ -z "${AS_OUTPUT_DIR}" ]]; then
  AS_OUTPUT_DIR="${RUN_DIR}/results/AS"
fi
if [[ -z "${AS_BASELINES_OUTPUT_DIR}" ]]; then
  AS_BASELINES_OUTPUT_DIR="${RUN_DIR}/results/baselines"
fi

mkdir -p "${RUN_DIR}" "${AS_OUTPUT_DIR}" "${AS_BASELINES_OUTPUT_DIR}" "${AS_CACHE_DIR}"

run_step() {
  if [[ -n "${SLURM_STEP_ID:-}" && "${SLURM_STEP_ID}" != "batch" ]]; then
    "$@"
  else
    srun --cpu-bind=cores "$@"
  fi
}

# Reproducibility snapshot
cp "${EXP_FILE}" "${RUN_DIR}/experiment.env"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null || true
python -m pip freeze > "${RUN_DIR}/pip_freeze.txt" || true
{
  echo "RUN_ID=${RUN_ID}"
  echo "JOB_ID=${SLURM_JOB_ID:-manual}"
  echo "NODE=$(hostname)"
  echo "STARTED=$(date -Is)"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "AS_DATA_FORMAT=${AS_DATA_FORMAT}"
  echo "AS_DATA_DIR=${AS_DATA_DIR}"
  echo "AS_CACHE_DIR=${AS_CACHE_DIR}"
  echo "AS_OUTPUT_DIR=${AS_OUTPUT_DIR}"
  echo "AS_BASELINES_OUTPUT_DIR=${AS_BASELINES_OUTPUT_DIR}"
  echo "AS_HORIZON_MODE=${AS_HORIZON_MODE}"
  echo "AS_WORKERS=${AS_WORKERS}"
  echo "AS_LOBSTER_LEVELS=${AS_LOBSTER_LEVELS}"
  echo "AS_RESUME_FROM_CALIBRATION=${AS_RESUME_FROM_CALIBRATION:-}"
  echo "EXP_FILE=${EXP_FILE}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-not set}"
} | tee "${RUN_DIR}/manifest.txt"

if [[ ! -d "${AS_DATA_DIR}" ]]; then
  echo "ERROR: AS data directory does not exist: ${AS_DATA_DIR}" >&2
  exit 2
fi
if ! find "${AS_DATA_DIR}" -maxdepth 2 -type f -name '*message_10.csv' | grep -q .; then
  echo "ERROR: no *_message_10.csv files found under ${AS_DATA_DIR}" >&2
  exit 2
fi
if ! find "${AS_DATA_DIR}" -maxdepth 2 -type f -name '*orderbook_10.csv' | grep -q .; then
  echo "ERROR: no *_orderbook_10.csv files found under ${AS_DATA_DIR}" >&2
  exit 2
fi
if [[ -n "${AS_RESUME_FROM_CALIBRATION:-}" && ! -f "${AS_RESUME_FROM_CALIBRATION}" ]]; then
  echo "ERROR: AS_RESUME_FROM_CALIBRATION does not exist: ${AS_RESUME_FROM_CALIBRATION}" >&2
  exit 2
fi

cat <<INFO
============================================================
AS classical-control baseline job
============================================================
Run ID:              ${RUN_ID}
Job ID:              ${SLURM_JOB_ID:-manual}
Node:                $(hostname)
Started:             $(date -Is)
Repo:                ${REPO_ROOT}
Data dir:            ${AS_DATA_DIR}
Cache dir:           ${AS_CACHE_DIR}
Output dir:          ${AS_OUTPUT_DIR}
Baselines output:    ${AS_BASELINES_OUTPUT_DIR}
Data format:         ${AS_DATA_FORMAT}
Horizon mode:        ${AS_HORIZON_MODE}
Workers:             ${AS_WORKERS}
LOBSTER levels:      ${AS_LOBSTER_LEVELS}
Order size:          ${AS_ORDER_SIZE}
Inventory limit:     ${AS_INVENTORY_LIMIT}
Decision interval:   ${AS_DECISION_INTERVAL_MS} ms
Episode steps:       ${AS_EPISODE_STEPS}
Include symmetric:   ${AS_INCLUDE_SYMMETRIC:-false}
Resume calibration:  ${AS_RESUME_FROM_CALIBRATION:-none}
Run dir:             ${RUN_DIR}
============================================================
INFO

if [[ "${AS_RUN_TESTS:-false}" == "true" ]]; then
  run_step python -m unittest discover -s as_model/tests
fi

CMD=(
  python -u as_model/script/run_as_baseline.py
  --data-format "${AS_DATA_FORMAT}"
  --data-dir "${AS_DATA_DIR}"
  --cache-dir "${AS_CACHE_DIR}"
  --output-dir "${AS_OUTPUT_DIR}"
  --baselines-output-dir "${AS_BASELINES_OUTPUT_DIR}"
  --workers "${AS_WORKERS}"
  --lobster-levels "${AS_LOBSTER_LEVELS}"
  --as-horizon-mode "${AS_HORIZON_MODE}"
  --order-size "${AS_ORDER_SIZE}"
  --inventory-limit "${AS_INVENTORY_LIMIT}"
  --decision-interval-ms "${AS_DECISION_INTERVAL_MS}"
  --episode-steps "${AS_EPISODE_STEPS}"
  --tick-size "${AS_TICK_SIZE}"
  --q-ref "${AS_Q_REF}"
  --target-skew "${AS_TARGET_SKEW}"
  --delta-start-ticks "${AS_DELTA_START_TICKS}"
  --delta-end-ticks "${AS_DELTA_END_TICKS}"
)

if [[ -n "${AS_RESUME_FROM_CALIBRATION:-}" ]]; then
  CMD+=(--resume-from-calibration "${AS_RESUME_FROM_CALIBRATION}")
fi
if [[ -n "${AS_GAMMA_VALUES:-}" ]]; then
  CMD+=(--gamma-values "${AS_GAMMA_VALUES}")
fi
if [[ -n "${AS_EPISODE_STARTS_CSV:-}" ]]; then
  CMD+=(--episode-starts-csv "${AS_EPISODE_STARTS_CSV}")
fi
if [[ "${AS_INCLUDE_SYMMETRIC:-false}" == "true" ]]; then
  CMD+=(--include-symmetric)
fi
if [[ "${AS_NO_PLOTS:-true}" == "true" ]]; then
  CMD+=(--no-plots)
fi
if [[ "${AS_DEBUG_PLOTS:-false}" == "true" ]]; then
  CMD+=(--debug-plots)
fi

printf 'Command:' | tee -a "${RUN_DIR}/manifest.txt"
printf ' %q' "${CMD[@]}" | tee -a "${RUN_DIR}/manifest.txt"
printf '\n' | tee -a "${RUN_DIR}/manifest.txt"

run_step "${CMD[@]}"

{
  echo "FINISHED=$(date -Is)"
  echo "AS_CALIBRATION=${AS_OUTPUT_DIR}/as_calibration.json"
  echo "AS_GAMMA_SELECTION=${AS_OUTPUT_DIR}/as_gamma_selection.csv"
  echo "AS_TEST_METRICS=${AS_OUTPUT_DIR}/as_test_metrics.csv"
  echo "AS_RL_COMPARISON=${AS_OUTPUT_DIR}/as_for_rl_comparison.csv"
  echo "AS_RUN_MANIFEST=${AS_OUTPUT_DIR}/as_run_manifest.json"
} | tee -a "${RUN_DIR}/manifest.txt"
