#!/usr/bin/env bash
# Submit a JaxMARL-HFT validation/evaluation experiment through Slurm.
# Usage:
#   bash snellius_scripts/submit_eval.sh snellius_scripts/experiments/h100x4_4096_8388608_val.env
set -euo pipefail

cd /home/adong/JaxMARL-HFT
source snellius_scripts/00_cluster.sh

EXP_FILE="${1:-}"
if [[ -z "${EXP_FILE}" ]]; then
  echo "Usage: bash snellius_scripts/submit_eval.sh <experiment.env>" >&2
  exit 1
fi
if [[ ! -f "${EXP_FILE}" ]]; then
  echo "ERROR: experiment file not found: ${EXP_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${EXP_FILE}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

# Resource options are passed here because #SBATCH lines are parsed before bash can source env files.
sbatch \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --nodes="${NODES}" \
  --ntasks="${NTASKS}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --gpus="${GPUS}" \
  --mem="${MEM}" \
  --time="${WALLTIME}" \
  --output="${LOG_DIR}/%x-%j.out" \
  --error="${LOG_DIR}/%x-%j.err" \
  --export=ALL,EXP_FILE="${EXP_FILE}" \
  snellius_scripts/jobs/run_eval.sbatch
