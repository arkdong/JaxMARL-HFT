# JaxMARL-HFT Snellius research pipeline

Install by copying `snellius_scripts/` into `/home/adong/JaxMARL-HFT/`.

Submit the 4-H100 4096-env experiment:

```bash
cd /home/adong/JaxMARL-HFT
bash snellius_scripts/submit_train.sh snellius_scripts/experiments/h100x4_4096_8388608.env
```

Run the AS classical-control baseline:

```bash
cd /home/adong/JaxMARL-HFT
bash snellius_scripts/submit_as_baseline.sh snellius_scripts/experiments/as_lobster_baseline.env
```

Run the same AS job interactively with `srun`:

```bash
cd /home/adong/JaxMARL-HFT
srun \
  --partition=rome \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=32 \
  --mem=240G \
  --time=12:00:00 \
  --export=ALL,EXP_FILE=snellius_scripts/experiments/as_lobster_baseline.env \
  bash snellius_scripts/jobs/run_as_baseline.sbatch
```

AS outputs are written by default to:

```text
runs/as_lobster_amzn_finite_<jobid>/results/AS/
```

To resume after calibration, edit the env file and set:

```bash
export AS_RESUME_FROM_CALIBRATION="/path/to/as_calibration.json"
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
- `submit_as_baseline.sh`: reads AS settings and passes Slurm resources to `sbatch`.
- `jobs/run_train.sbatch`: generic training body, not experiment-specific.
- `jobs/run_as_baseline.sbatch`: AS calibration, validation gamma selection, and test evaluation body.
- `runs/<experiment>_<jobid>/`: copied experiment file, git commit, pip freeze, manifest.
