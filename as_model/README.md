# LOBSTER AS Baseline Bundle

This folder is a portable copy of the code needed to run the cleaned
Avellaneda-Stoikov baseline on AMZN LOBSTER 10-level data.

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
  --workers 32
```

Or:

```bash
bash run_lobster_as.sh
```

Outputs are written to:

```text
results/AS/
results/baselines/
```

## Smoke Test

The tiny fixture is included only to verify the code runs:

```bash
python -m unittest discover -s tests
```

