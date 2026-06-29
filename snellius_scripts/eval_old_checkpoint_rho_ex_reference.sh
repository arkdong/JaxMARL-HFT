#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_ROOT:-/home/adong/JaxMARL-HFT}"

EXP_FILE="${1:-snellius_scripts/eval/rho_ex_reference_validation_smoke.env}"
if [[ "${EXP_FILE}" == *=* ]]; then
  cat >&2 <<'MSG'
ERROR: eval_old_checkpoint_rho_ex_reference.sh now submits the standard Snellius eval job.
Put overrides in snellius_scripts/eval/rho_ex_reference_validation_smoke.env, or pass
another eval env file as the first argument.

Example:
  bash snellius_scripts/submit_eval.sh snellius_scripts/eval/rho_ex_reference_validation_smoke.env
MSG
  exit 2
fi

if [[ "$#" -gt 1 ]]; then
  echo "ERROR: expected at most one argument: an eval .env file." >&2
  exit 2
fi

exec bash snellius_scripts/submit_eval.sh "${EXP_FILE}"
