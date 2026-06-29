# Classical Models

This folder contains the classical baselines used for comparison with the
JaxMARL-HFT agents. It is intentionally isolated from the root JaxMARL project:
AC and AS share this folder's `uv` project and virtual environment, and the
root requirements files are left unchanged.

## Layout

```text
classical_models/
  pyproject.toml
  uv.lock
  .venv/
  ac_model/
  as_model/
```

- `ac_model/`: Almgren-Chriss execution benchmark.
- `as_model/`: Avellaneda-Stoikov market-making baseline.

## Setup

```bash
cd classical_models
uv sync --extra dev
```

## Run AC

```bash
cd classical_models
uv run ac-benchmark --help
bash ac_model/scripts/run_ac_benchmark.sh
```

## Run AS

```bash
cd classical_models
uv run python as_model/script/run_as_baseline.py --help
uv run python -m unittest discover -s as_model/tests
```

From the repository root you can also run:

```bash
bash classical_models/as_model/run_lobster_as.sh
```
