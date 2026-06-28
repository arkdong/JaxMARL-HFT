# JaxMARL-HFT Snellius research pipeline

Install by copying `snellius_scripts/` into `/home/adong/JaxMARL-HFT/`.

Submit the 4-H100 4096-env experiment:

```bash
cd /home/adong/JaxMARL-HFT
bash snellius_scripts/submit_train.sh snellius_scripts/experiments/h100x4_4096_8388608.env
```

Monitor:

```bash
bash snellius_scripts/tools/monitor.sh queue
bash snellius_scripts/tools/monitor.sh latest
```

Design:

- `00_cluster.sh`: stable Snellius/user settings only.
- `experiments/*.env`: one file per reproducible experiment.
- `submit_train.sh`: reads experiment settings and passes Slurm resources to `sbatch`.
- `jobs/run_train.sbatch`: generic training body, not experiment-specific.
- `runs/<experiment>_<jobid>/`: copied experiment file, git commit, pip freeze, manifest.
