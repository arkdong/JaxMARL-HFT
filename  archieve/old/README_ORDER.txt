Run order on Snellius, assuming your repo is /home/adong/JaxMARL-HFT:

0) Copy this directory into the repo:
   cp -r snellius_jaxmarl_scripts /home/adong/JaxMARL-HFT/snellius_scripts
   cd /home/adong/JaxMARL-HFT

1) Optional but recommended: check available Python module:
   module load 2025
   module avail Python/3.12
   # If auto-detection fails later, edit snellius_scripts/00_config.sh or run:
   export PYTHON_MODULE="Python/3.12.x-GCCcore-yy.y.0"

2) Prepare folders and check quota:
   bash snellius_scripts/01_prepare_home_layout.sh

3) Upload LOBSTER data from your LOCAL computer, or place files manually under:
   /home/adong/JaxMARL-HFT/data/rawLOBSTER/AMZN/2024/
   Use snellius_scripts/local_upload_lobster_template.sh as a template on your LOCAL machine.

4) Install Python environment:
   bash snellius_scripts/02_setup_python_env.sh

5) Patch JaxMARL-HFT configs:
   bash snellius_scripts/03_configure_jaxmarl_hft.sh

6) Submit a GPU/JAX sanity check:
   sbatch snellius_scripts/04_gpu_jax_test.sbatch
   bash snellius_scripts/07_monitoring_commands.sh queue

7) Submit a small training smoke test:
   sbatch snellius_scripts/05_smoke_train.sbatch

8) Submit the longer A100 run only after the smoke test succeeds:
   sbatch snellius_scripts/06_train_a100.sbatch

Increase TRAIN_NUM_ENVS in 00_config.sh gradually: 64 -> 256 -> 512 -> 1024 -> 2048.
