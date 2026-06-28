#!/usr/bin/env bash
set -u   # 允许未定义变量报错
set -o pipefail

# ---------------------- 0) Project workspace ----------------------
export PRJROOT="/projects/prjs1859"
export USERROOT="${PRJROOT}/users/${USER}"
mkdir -p "${USERROOT}"

# ---------------------- 1) (Optional) CUDA module ----------------------
# Snellius 上具体 module 名称你确认后再改，比如：
module load 2024
module load CUDA/12.6.0
# 或者 module load cuda/11.x
#
# module load <CUDA_MODULE_HERE>

# ---------------------- 2) Conda init & activate ----------------------
# 你的 miniconda 安装在：/home/xzhang4/miniconda3
# source "${HOME}/miniconda3/bin/activate"
# conda activate py310
eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
conda activate py310

# ---------------------- 3) Put caches/configs into /projects ----------------------
# Matplotlib config (avoid writing to $HOME)
export MPLCONFIGDIR="${USERROOT}/.config/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

# HuggingFace / Transformers cache
export HF_HOME="${USERROOT}/.cache/huggingface"
# `TRANSFORMERS_CACHE` is deprecated in Transformers v5; use `HF_HOME` instead.
# Unset it explicitly in case it is inherited from parent shells.
unset TRANSFORMERS_CACHE
mkdir -p "${HF_HOME}"

# General cache home (many libs respect this)
export XDG_CACHE_HOME="${USERROOT}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

# Pip cache (optional but recommended)
export PIP_CACHE_DIR="${USERROOT}/.cache/pip"
mkdir -p "${PIP_CACHE_DIR}"

# Torch hub / checkpoints cache (optional)
export TORCH_HOME="${USERROOT}/.cache/torch"
mkdir -p "${TORCH_HOME}"

echo "[OK] Activated conda env: $(which python) ($(python --version))"
echo "[OK] Caches redirected under: ${USERROOT}"
