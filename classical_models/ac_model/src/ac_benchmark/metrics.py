from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_METRIC_COLUMNS = [
    "slippage_ticks_per_share",
    "slippage_per_share_task",
    "slippage_notional",
    "avg_sq_remaining",
    "completion_step",
    "half_completion_step",
    "unfinished",
    "completion_rate",
    "n_fill_steps",
]


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 7,
) -> Tuple[float, float, float]:
    """Return mean and percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1 or n_boot <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(int(n_boot), len(arr)))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return mean, float(lo), float(hi)


def aggregate_by_policy(
    metrics_df: pd.DataFrame,
    *,
    group_col: str = "kappa_T",
    metric_cols: Sequence[str] = DEFAULT_METRIC_COLUMNS,
    n_boot: int = 2000,
    seed: int = 7,
) -> pd.DataFrame:
    """Aggregate episode-level metrics by kappa_T policy."""
    records = []
    for group_value, g in metrics_df.groupby(group_col, sort=True):
        rec = {group_col: group_value, "n_episodes": int(g["episode_id"].nunique())}
        for col in metric_cols:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
            rec[f"{col}_mean"] = mean
            rec[f"{col}_median"] = float(np.median(vals))
            rec[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rec[f"{col}_ci95_lo"] = lo
            rec[f"{col}_ci95_hi"] = hi
        records.append(rec)
    return pd.DataFrame.from_records(records)


def paired_policy_difference(
    metrics_df: pd.DataFrame,
    *,
    metric: str,
    baseline_kappa_T: float,
    compare_kappa_T: float,
    n_boot: int = 2000,
    seed: int = 7,
) -> dict:
    """Compute paired difference compare-baseline over common episode IDs."""
    base = metrics_df[metrics_df["kappa_T"] == baseline_kappa_T][["episode_id", metric]].rename(columns={metric: "baseline"})
    comp = metrics_df[metrics_df["kappa_T"] == compare_kappa_T][["episode_id", metric]].rename(columns={metric: "compare"})
    merged = pd.merge(base, comp, on="episode_id", how="inner")
    diffs = pd.to_numeric(merged["compare"], errors="coerce") - pd.to_numeric(merged["baseline"], errors="coerce")
    mean, lo, hi = bootstrap_mean_ci(diffs, n_boot=n_boot, seed=seed)
    return {
        "metric": metric,
        "baseline_kappa_T": float(baseline_kappa_T),
        "compare_kappa_T": float(compare_kappa_T),
        "n_pairs": int(len(diffs)),
        "mean_diff": float(mean),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
    }


def add_policy_names(df: pd.DataFrame) -> pd.DataFrame:
    """Attach readable policy labels."""
    out = df.copy()

    def name(k: float) -> str:
        k = float(k)
        if abs(k) < 1e-12:
            return "TWAP / AC kappaT=0"
        if k <= 0.5:
            return f"AC-Slow kappaT={k:g}"
        if k <= 1.0:
            return f"AC-Medium kappaT={k:g}"
        if k <= 2.0:
            return f"AC-Fast kappaT={k:g}"
        return f"AC-VeryFast kappaT={k:g}"

    out["policy"] = out["kappa_T"].map(name)
    return out

