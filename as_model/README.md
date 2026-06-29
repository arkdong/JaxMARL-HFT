# LOBSTER AS Classical-Control Baseline

This folder is a portable copy of the code needed to run the cleaned
Avellaneda-Stoikov classical-control baseline on AMZN LOBSTER 10-level data.
The main pipeline calibrates `sigma`, `A`, and `k` on train data, selects one
`gamma*` on validation data, and evaluates the fixed finite-horizon AS policy
on held-out test episodes.

## Folder Layout

Expected data layout:

```text
data/lobster_amzn_10/
  train/
    AMZN_YYYY-MM-DD_34200000_57600000_message_10.csv
    AMZN_YYYY-MM-DD_34200000_57600000_orderbook_10.csv
  val/
    ...
  test/
    ...
```

`val/` and `validation/` are both accepted.

The bundle does not include the real LOBSTER data. Copy your CSV folders into
`data/lobster_amzn_10/` on the target machine.

## Setup

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With uv:

```bash
uv sync
```

## Run

```bash
python script/run_as_baseline.py \
  --data-format lobster \
  --data-dir data/lobster_amzn_10 \
  --cache-dir data/cache/as_lobster_10 \
  --as-horizon-mode finite_episode \
  --workers 32
```

Or:

```bash
bash run_lobster_as.sh
```

Outputs are written to:

```text
results/AS/as_calibration.json
results/AS/as_gamma_selection.csv
results/AS/as_test_metrics.csv
results/AS/as_for_rl_comparison.csv
results/AS/as_run_manifest.json
```

Use `--resume-from-calibration results/AS/as_calibration.json` to skip train
calibration and continue with validation gamma selection and test evaluation.
Use `--include-symmetric` only when the old symmetric market-maker row is
needed as an auxiliary benchmark.

## Smoke Test

The tests generate tiny synthetic fixtures to verify the code runs:

```bash
python -m unittest discover -s tests
```
