from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import infer_depth, level_columns
from .schema import ACParams


def estimate_volatility_step(
    snapshots: pd.DataFrame,
    *,
    step_stride_rows: int = 1,
    winsor_quantile: float = 0.001,
) -> float:
    """Estimate per-environment-step mid-price volatility."""
    mids = pd.to_numeric(snapshots["mid_price"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(mids) <= step_stride_rows:
        raise ValueError("not enough rows to estimate volatility")
    sampled = mids[:: max(1, int(step_stride_rows))]
    diffs = np.diff(sampled)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        raise ValueError("not enough mid-price differences to estimate volatility")
    if winsor_quantile and 0 < winsor_quantile < 0.5:
        lo, hi = np.quantile(diffs, [winsor_quantile, 1.0 - winsor_quantile])
        diffs = np.clip(diffs, lo, hi)
    return float(np.std(diffs, ddof=1))


def estimate_half_spread(snapshots: pd.DataFrame) -> float:
    spreads = pd.to_numeric(snapshots["spread"], errors="coerce").dropna().to_numpy(dtype=float)
    spreads = spreads[np.isfinite(spreads) & (spreads >= 0)]
    if len(spreads) == 0:
        raise ValueError("no valid spreads found")
    return float(np.median(spreads) / 2.0)


def book_walk_vwap(row: pd.Series, side: str, quantity: int, depth: int) -> Tuple[int, float, float]:
    """Walk top-depth book and return executed quantity, VWAP, notional.

    side='buy' consumes asks, side='sell' consumes bids.
    """
    if quantity <= 0:
        return 0, float("nan"), 0.0
    price_cols, size_cols = level_columns(side, depth)
    prices = pd.to_numeric(row[price_cols], errors="coerce").to_numpy(dtype=float)
    sizes = pd.to_numeric(row[size_cols], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(prices) & np.isfinite(sizes) & (prices > 0) & (sizes > 0)
    prices = prices[mask]
    sizes = sizes[mask]
    if len(prices) == 0:
        return 0, float("nan"), 0.0
    order = np.argsort(prices) if side == "buy" else np.argsort(-prices)
    prices = prices[order]
    sizes = sizes[order]

    remaining = int(quantity)
    executed = 0
    notional = 0.0
    for price, size in zip(prices, sizes):
        take = min(remaining, int(size))
        if take <= 0:
            continue
        notional += take * float(price)
        executed += take
        remaining -= take
        if remaining <= 0:
            break
    if executed == 0:
        return 0, float("nan"), 0.0
    return executed, notional / executed, notional


def estimate_temporary_impact_eta(
    snapshots: pd.DataFrame,
    *,
    q_grid: Sequence[int] = (10, 20, 50, 100, 200, 400, 600),
    depth: Optional[int] = None,
    max_rows: Optional[int] = 250_000,
    random_seed: int = 7,
    half_spread: Optional[float] = None,
) -> Tuple[float, pd.DataFrame]:
    """Estimate AC temporary-impact slope eta from book-walk impact.

    For each sampled snapshot and size q, we compute immediate book-walk impact
    relative to the mid price, subtract the median half-spread epsilon, and use
    robust median slopes (impact-epsilon)/q. The returned eta is in dollars per
    share squared per step under the step-normalised benchmark.
    """
    if depth is None:
        depth = infer_depth(snapshots)
    if depth <= 0:
        raise ValueError("depth must be positive and snapshots must contain level prices/sizes")
    if half_spread is None:
        half_spread = estimate_half_spread(snapshots)

    q_grid = [int(q) for q in q_grid if int(q) > 0]
    if not q_grid:
        raise ValueError("q_grid must contain positive quantities")

    df = snapshots.reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        rng = np.random.default_rng(random_seed)
        idx = np.sort(rng.choice(len(df), size=int(max_rows), replace=False))
        df = df.iloc[idx].reset_index(drop=True)

    records = []
    slope_values = []
    for q in q_grid:
        for side in ["buy", "sell"]:
            impacts = []
            complete = 0
            for _, row in df.iterrows():
                executed, vwap, _ = book_walk_vwap(row, side, q, depth)
                if executed < q or not np.isfinite(vwap):
                    continue
                mid = float(row["mid_price"])
                impact = (vwap - mid) if side == "buy" else (mid - vwap)
                if np.isfinite(impact):
                    impacts.append(float(impact))
                    complete += 1
            if impacts:
                impacts_arr = np.asarray(impacts, dtype=float)
                median_impact = float(np.median(impacts_arr))
                p90_impact = float(np.quantile(impacts_arr, 0.90))
                slope = max(0.0, (median_impact - float(half_spread)) / float(q))
                slope_values.append(slope)
                records.append(
                    {
                        "q": int(q),
                        "side": side,
                        "n_complete": int(complete),
                        "median_impact": median_impact,
                        "p90_impact": p90_impact,
                        "half_spread": float(half_spread),
                        "eta_slope": float(slope),
                    }
                )
            else:
                records.append(
                    {
                        "q": int(q),
                        "side": side,
                        "n_complete": 0,
                        "median_impact": np.nan,
                        "p90_impact": np.nan,
                        "half_spread": float(half_spread),
                        "eta_slope": np.nan,
                    }
                )

    impact_curve = pd.DataFrame.from_records(records)
    valid_slopes = np.asarray([s for s in slope_values if np.isfinite(s) and s >= 0], dtype=float)
    eta = float(np.median(valid_slopes)) if len(valid_slopes) else 0.0
    return eta, impact_curve


def calibrate_ac_params(
    snapshots: pd.DataFrame,
    *,
    q_grid: Sequence[int] = (10, 20, 50, 100, 200, 400, 600),
    step_stride_rows: int = 1,
    depth: Optional[int] = None,
    max_rows: Optional[int] = 250_000,
    random_seed: int = 7,
) -> Tuple[ACParams, pd.DataFrame]:
    sigma_step = estimate_volatility_step(snapshots, step_stride_rows=step_stride_rows)
    half_spread = estimate_half_spread(snapshots)
    eta_ac, impact_curve = estimate_temporary_impact_eta(
        snapshots,
        q_grid=q_grid,
        depth=depth,
        max_rows=max_rows,
        random_seed=random_seed,
        half_spread=half_spread,
    )
    params = ACParams(
        sigma_step=float(sigma_step),
        half_spread=float(half_spread),
        eta_ac=float(eta_ac),
        gamma_ac=0.0,
        q_grid=list(map(int, q_grid)),
        n_obs=int(len(snapshots)),
    )
    return params, impact_curve


def save_calibration(params: ACParams, impact_curve: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ac_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(params.to_dict(), f, indent=2)
    impact_curve.to_csv(out / "book_walk_impact_curve.csv", index=False)


def load_calibration(path: str | Path) -> ACParams:
    with Path(path).open("r", encoding="utf-8") as f:
        return ACParams.from_dict(json.load(f))


def theoretical_cost_variance(
    schedule: Sequence[int],
    *,
    sigma_step: float,
    epsilon: float,
    eta: float,
    gamma: float = 0.0,
) -> dict:
    """Compute step-normalised AC E and V for a fixed schedule.

    This diagnostic uses tau=1 environment step. It is for comparing schedule
    shape and calibration scale, not for the realised replay metric.
    """
    n = np.asarray(schedule, dtype=float)
    X = float(n.sum())
    remaining = X - np.cumsum(n)
    E = 0.5 * float(gamma) * X * X + float(epsilon) * float(np.sum(np.abs(n))) + float(eta) * float(np.sum(n * n))
    V = float(sigma_step) ** 2 * float(np.sum(remaining * remaining))
    return {"expected_cost": float(E), "variance": float(V), "std_cost": float(np.sqrt(max(V, 0.0)))}

