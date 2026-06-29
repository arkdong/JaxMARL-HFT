from __future__ import annotations

"""Vectorized calibration helpers for the AC benchmark."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import estimate_half_spread, estimate_volatility_step
from .data import infer_depth
from .fast_replay import _book_walk_batch, _depth_arrays
from .schema import ACParams


def estimate_temporary_impact_eta_fast(
    snapshots: pd.DataFrame,
    *,
    q_grid: Sequence[int] = (10, 20, 50, 100, 200, 400, 600),
    depth: Optional[int] = None,
    max_rows: Optional[int] = 250_000,
    random_seed: int = 7,
    half_spread: Optional[float] = None,
) -> Tuple[float, pd.DataFrame]:
    """Estimate temporary-impact slope with batched book walking."""
    if depth is None:
        depth = infer_depth(snapshots)
    if depth <= 0:
        raise ValueError("depth must be positive")
    if half_spread is None:
        half_spread = estimate_half_spread(snapshots)

    q_values = [int(q) for q in q_grid if int(q) > 0]
    if not q_values:
        raise ValueError("q_grid must contain positive quantities")

    df = snapshots.reset_index(drop=True)
    if max_rows is not None and len(df) > int(max_rows):
        rng = np.random.default_rng(random_seed)
        sample_idx = np.sort(rng.choice(len(df), size=int(max_rows), replace=False))
        df = df.iloc[sample_idx].reset_index(drop=True)

    arrays = _depth_arrays(df, depth, dtype=np.float64)
    mids = arrays["mid"]
    records: List[dict] = []
    slope_values: List[float] = []

    for q in q_values:
        qty = np.full(len(df), q, dtype=np.float64)
        for side, price_key, size_key in (
            ("buy", "ask_prices", "ask_sizes"),
            ("sell", "bid_prices", "bid_sizes"),
        ):
            executed, vwap, _ = _book_walk_batch(arrays[price_key], arrays[size_key], qty)
            complete = (executed >= q) & np.isfinite(vwap)
            impact = vwap - mids if side == "buy" else mids - vwap
            impact = impact[complete & np.isfinite(impact)]

            if len(impact):
                median_impact = float(np.median(impact))
                p90_impact = float(np.quantile(impact, 0.90))
                slope = max(0.0, (median_impact - float(half_spread)) / float(q))
                slope_values.append(slope)
            else:
                median_impact = np.nan
                p90_impact = np.nan
                slope = np.nan

            records.append(
                {
                    "q": int(q),
                    "side": side,
                    "n_complete": int(len(impact)),
                    "median_impact": median_impact,
                    "p90_impact": p90_impact,
                    "half_spread": float(half_spread),
                    "eta_slope": slope,
                }
            )

    valid_slopes = np.asarray([s for s in slope_values if np.isfinite(s) and s >= 0.0], dtype=float)
    eta = float(np.median(valid_slopes)) if len(valid_slopes) else 0.0
    return eta, pd.DataFrame.from_records(records)


def calibrate_ac_params_fast(
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
    eta_ac, impact_curve = estimate_temporary_impact_eta_fast(
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
