#!/usr/bin/env bash
#SBATCH --job-name=rho_ex_ref_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/rho_ex_ref_eval_%j.out
#SBATCH --error=slurm_logs/rho_ex_ref_eval_%j.err

set -euo pipefail

cd "${REPO_ROOT:-/home/adong/JaxMARL-HFT}"
mkdir -p slurm_logs

export EXP_FILE="${EXP_FILE:-snellius_scripts/eval/rho_ex_reference_validation_smoke.env}"
exec bash snellius_scripts/jobs/run_eval.sbatch
