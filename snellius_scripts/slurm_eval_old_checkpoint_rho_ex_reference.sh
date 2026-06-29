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

REPO_DIR="${REPO_DIR:-/gpfs/home4/adong/JaxMARL-HFT}"
cd "${REPO_DIR}"
mkdir -p slurm_logs

# Uncomment/adapt if your cluster needs conda/module setup:
# source ~/.bashrc
# conda activate jaxmarl_hft
# module load cuda

bash scripts/eval_old_checkpoint_rho_ex_reference.sh "$@"
