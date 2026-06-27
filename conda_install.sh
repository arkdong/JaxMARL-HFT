wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
echo "[OK] Download conda env: $($HOME/miniconda3/bin/conda --version)"