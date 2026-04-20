#!/usr/bin/env bash
set -u
set -o pipefail

module purge
module load 2024
module load Miniconda3/25.5.1-1

eval "$(conda shell.bash hook)"

conda create -n jaxmarl_hft python=3.10 -y
conda activate jaxmarl_hft

pip install "jax[cuda12]"
pip install -r requirements.txt