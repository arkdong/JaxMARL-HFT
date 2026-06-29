#!/usr/bin/env bash
# Submit the AS classical-control baseline through Slurm.
# Usage:
#   bash snellius_scripts/submit_as_baseline.sh snellius_scripts/experiments/as_lobster_baseline.env
set -euo pipefail

cd /home/adong/JaxMARL-HFT
source snellius_scripts/00_cluster.sh

EXP_FILE="${1:-}"
if [[ -z "${EXP_FILE}" ]]; then
  echo "Usage: bash snellius_scripts/submit_as_baseline.sh <experiment.env>" >&2
  exit 1
fi
if [[ ! -f "${EXP_FILE}" ]]; then
  echo "ERROR: experiment file not found: ${EXP_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${EXP_FILE}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

SBATCH_ARGS=(
  --job-name="${JOB_NAME}"
  --partition="${PARTITION}"
  --nodes="${NODES}"
  --ntasks="${NTASKS}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM}"
  --time="${WALLTIME}"
  --output="${LOG_DIR}/%x-%j.out"
  --error="${LOG_DIR}/%x-%j.err"
  --export=ALL,EXP_FILE="${EXP_FILE}"
)

if [[ "${GPUS:-0}" != "0" && -n "${GPUS:-}" ]]; then
  SBATCH_ARGS+=(--gpus="${GPUS}")
fi

sbatch "${SBATCH_ARGS[@]}" snellius_scripts/jobs/run_as_baseline.sbatch
