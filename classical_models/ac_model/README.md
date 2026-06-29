# Almgren-Chriss Execution Benchmark for the Thesis

This package implements a small, deterministic Almgren-Chriss (AC) benchmark for the execution agent. It is designed to produce a classical comparison dataset that can later be evaluated on the same AMZN episodes as the JaxMARL-HFT execution policies.

The benchmark does **not** train a model and does **not** change the thesis reward definitions. It only creates fixed AC/TWAP execution schedules and replays them against historical LOB snapshots.

## What is implemented

- Discrete AC schedule with normalized `kappa_T` grid:
  - `kappa_T = 0`: TWAP / risk-neutral AC limit
  - larger `kappa_T`: more front-loaded execution
- Canonical snapshot schema for LOBSTER-style and Databento-like top-K book snapshots.
- LOBSTER orderbook CSV loader.
- Generic snapshot standardizer for Databento/MBP-style exports that already contain top-K bid/ask price and size columns.
- Training calibration diagnostics:
  - per-step mid-price volatility
  - median half-spread
  - temporary-impact slope from book-walking
- Deterministic replay:
  - buy consumes ask depth
  - sell consumes bid depth
  - scheduled quantities are carried forward if top-K depth is insufficient before the final step
- Thesis metrics:
  - slippage in dollars, per share, and ticks/share
  - unfinished quantity
  - average squared remaining quantity
  - completion step
  - fill steps
  - paired differences versus TWAP
- Plots:
  - AC schedules
  - book-walk impact curve
  - risk-cost frontier
  - slippage distributions

## Install

```bash
cd classical_models
uv sync --extra dev
```

## Canonical snapshot schema

The evaluator expects one row per book snapshot or message-indexed book state. Required columns:

```text
timestamp
best_bid
best_ask
mid_price
spread
bid_price_1, bid_size_1, ask_price_1, ask_size_1
...
bid_price_K, bid_size_K, ask_price_K, ask_size_K
```

If the file has flexible names such as `bid_px_00`, `ask_px_00`, `bid_sz_00`, `ask_sz_00`, use `standardize-snapshots` first.

## LOBSTER input

For a LOBSTER orderbook file with no header:

```bash
uv run ac-benchmark normalize-lobster \
  --orderbook AMZN_orderbook_10.csv \
  --messages AMZN_message.csv \
  --levels 10 \
  --price-scale 10000 \
  --out data/amzn_train.parquet
```

Run the same command for validation and test files.

## Databento-like input

Fast path: if your Databento export already contains top-K book snapshots or MBP-like rows, standardize it directly:

```bash
uv run ac-benchmark standardize-snapshots \
  --input databento_amzn_train_top10.csv \
  --levels 10 \
  --out data/amzn_train.parquet
```

Raw MBO fallback: if your file contains Databento-like order-level events with columns similar to `timestamp/ts_event`, `action`, `side`, `price`, `size`, and `order_id`, reconstruct top-K snapshots first:

```bash
uv run ac-benchmark reconstruct-mbo \
  --input databento_amzn_train_mbo.csv \
  --levels 10 \
  --sample-every-events 100 \
  --out data/amzn_train.parquet
```

The raw-MBO reconstructor is a simple CPU fallback for thesis benchmarking. For very large months of AMZN MBO data, a vendor/exported MBP snapshot file or an existing JAX-LOB preprocessing pipeline will usually be faster. Use `--price-scale` only if your export stores integer-scaled prices.

## Calibration on the 3-month training split

```bash
uv run ac-benchmark calibrate \
  --snapshots data/amzn_train.parquet \
  --out-dir outputs/ac/train_calibration \
  --levels 10 \
  --messages-per-step 100 \
  --q-grid 10,20,50,100,200,400,600
```

If your snapshot file is already sampled once per environment step, set `--messages-per-step 1`.

## Validation sweep on the 1-month validation split

```bash
uv run ac-benchmark evaluate \
  --snapshots data/amzn_valid.parquet \
  --out-dir outputs/ac/validation_sweep \
  --levels 10 \
  --task-size 600 \
  --episode-length 64 \
  --messages-per-step 100 \
  --episode-start-frequency-steps 64 \
  --lot-size 10 \
  --tick-size 0.01 \
  --directions random \
  --kappa-grid 0,0.5,1,2,4
```

The main result files are:

```text
outputs/ac/validation_sweep/episode_metrics.csv
outputs/ac/validation_sweep/policy_summary.csv
outputs/ac/validation_sweep/paired_slippage_vs_twap.csv
outputs/ac/validation_sweep/risk_cost_frontier.png
outputs/ac/validation_sweep/ac_schedules.png
outputs/ac/validation_sweep/slippage_distribution.png
```

## Final held-out test benchmark

After selecting the AC policy or policies from validation, run the same command on the held-out test month. You can still run the full grid for transparency:

```bash
uv run ac-benchmark evaluate \
  --snapshots data/amzn_test.parquet \
  --out-dir outputs/ac/test_benchmark \
  --levels 10 \
  --task-size 600 \
  --episode-length 64 \
  --messages-per-step 100 \
  --episode-start-frequency-steps 64 \
  --lot-size 10 \
  --tick-size 0.01 \
  --directions random \
  --kappa-grid 0,1,2
```

## Smoke test with synthetic data

```bash
uv run ac-benchmark make-synthetic --out examples/synthetic.parquet --rows 10000 --levels 10
uv run ac-benchmark calibrate --snapshots examples/synthetic.parquet --out-dir outputs/ac/synthetic_calibration --levels 10 --messages-per-step 1
uv run ac-benchmark evaluate --snapshots examples/synthetic.parquet --out-dir outputs/ac/synthetic_eval --levels 10 --messages-per-step 1 --max-episodes 10
```

## Python API example

```python
from ac_benchmark.data import read_snapshots
from ac_benchmark.schema import EpisodeSpec
from ac_benchmark.replay import evaluate_ac_grid
from ac_benchmark.metrics import aggregate_by_policy

snapshots = read_snapshots("data/amzn_valid.parquet", depth_levels=10)
spec = EpisodeSpec(
    task_size=600,
    episode_length=64,
    messages_per_step=100,
    episode_start_frequency_steps=64,
    lot_size=10,
    tick_size=0.01,
    directions="random",
    random_seed=7,
)
metrics, fills, plan = evaluate_ac_grid(
    snapshots,
    spec=spec,
    kappa_T_grid=[0, 0.5, 1, 2, 4],
    depth=10,
    return_fills=False,
)
summary = aggregate_by_policy(metrics)
print(summary)
```

## Notes for later JaxMARL comparison

Use the same episode starts, directions, task size, episode length, step definition, and initial mid-price reference for AC and RL. The safest comparison is paired by `episode_id`:

```text
Delta slippage = slippage_RL - slippage_AC
```

Report the paired mean and bootstrap confidence interval.
