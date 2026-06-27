#!/usr/bin/env bash
set -u
set -o pipefail

# module purge
# module load 2025
# # module load Miniconda3/25.5.1-1

# eval "$(conda shell.bash hook)"

# conda create -n jaxmarl_hft python=3.10 -y
# conda activate jaxmarl_hft

# pip install "jax[cuda12]"
# pip install -r requirements.txt

# echo "[OK] Activated conda env: $(which python) ($(python --version))"
# echo "[OK] Pip: $(which pip)"

module purge
module load 2025

eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
# conda create -n jaxmarl_hft python=3.10

conda activate jaxmarl_hft

pip install "jax[cuda12]"
pip install -r requirements.txt