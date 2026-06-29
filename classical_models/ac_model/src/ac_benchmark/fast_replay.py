from __future__ import annotations

"""Vectorized Almgren-Chriss replay evaluator.

The original replay path is intentionally simple and fill-auditable, but it
uses pandas row access inside nested episode/step/kappa loops. This module keeps
the same book-walking semantics while evaluating all episodes for one file/day
as NumPy batches.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ac_policy import ac_schedule
from .data import build_episode_plan, infer_depth
from .schema import EpisodeSpec


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required snapshot columns: {missing}")


def _depth_arrays(
    snapshots: pd.DataFrame,
    depth: int,
    *,
    dtype: np.dtype | type = np.float64,
) -> Dict[str, np.ndarray]:
    """Extract top-K book columns into sorted, contiguous NumPy arrays."""
    level_cols: List[str] = []
    for lvl in range(1, depth + 1):
        level_cols.extend(
            [
                f"bid_price_{lvl}",
                f"bid_size_{lvl}",
                f"ask_price_{lvl}",
                f"ask_size_{lvl}",
            ]
        )
    _require_columns(snapshots, ["mid_price"] + level_cols)

    bid_prices = np.column_stack(
        [
            pd.to_numeric(snapshots[f"bid_price_{lvl}"], errors="coerce").to_numpy(dtype=dtype)
            for lvl in range(1, depth + 1)
        ]
    )
    bid_sizes = np.column_stack(
        [
            pd.to_numeric(snapshots[f"bid_size_{lvl}"], errors="coerce").to_numpy(dtype=dtype)
            for lvl in range(1, depth + 1)
        ]
    )
    ask_prices = np.column_stack(
        [
            pd.to_numeric(snapshots[f"ask_price_{lvl}"], errors="coerce").to_numpy(dtype=dtype)
            for lvl in range(1, depth + 1)
        ]
    )
    ask_sizes = np.column_stack(
        [
            pd.to_numeric(snapshots[f"ask_size_{lvl}"], errors="coerce").to_numpy(dtype=dtype)
            for lvl in range(1, depth + 1)
        ]
    )

    for prices, sizes in ((bid_prices, bid_sizes), (ask_prices, ask_sizes)):
        invalid = (~np.isfinite(prices)) | (~np.isfinite(sizes)) | (prices <= 0.0) | (sizes <= 0.0)
        sizes[~invalid] = np.floor(sizes[~invalid])
        prices[invalid] = 0.0
        sizes[invalid] = 0.0

    bid_order = np.argsort(-bid_prices, axis=1)
    ask_order = np.argsort(ask_prices + (ask_prices <= 0.0) * 1.0e18, axis=1)

    bid_prices = np.take_along_axis(bid_prices, bid_order, axis=1)
    bid_sizes = np.take_along_axis(bid_sizes, bid_order, axis=1)
    ask_prices = np.take_along_axis(ask_prices, ask_order, axis=1)
    ask_sizes = np.take_along_axis(ask_sizes, ask_order, axis=1)

    return {
        "mid": pd.to_numeric(snapshots["mid_price"], errors="coerce").to_numpy(dtype=dtype),
        "timestamp": snapshots["timestamp"].to_numpy() if "timestamp" in snapshots.columns else np.arange(len(snapshots)),
        "bid_prices": np.ascontiguousarray(bid_prices),
        "bid_sizes": np.ascontiguousarray(bid_sizes),
        "ask_prices": np.ascontiguousarray(ask_prices),
        "ask_sizes": np.ascontiguousarray(ask_sizes),
    }


def _book_walk_batch(
    prices: np.ndarray,
    sizes: np.ndarray,
    qty: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk displayed depth for many marketable orders at once."""
    if prices.shape != sizes.shape:
        raise ValueError("prices and sizes must have the same shape")

    q = np.maximum(np.asarray(qty, dtype=np.float64), 0.0)
    if prices.shape[0] != q.shape[0]:
        raise ValueError("qty must have one value per price/size row")

    cumulative = np.cumsum(sizes, axis=1)
    previous = np.empty_like(cumulative)
    previous[:, 0] = 0.0
    previous[:, 1:] = cumulative[:, :-1]

    take = np.minimum(np.maximum(q[:, None] - previous, 0.0), sizes)
    notional = np.sum(take * prices, axis=1)
    executed = np.sum(take, axis=1)
    vwap = np.divide(
        notional,
        executed,
        out=np.full(notional.shape, np.nan, dtype=np.float64),
        where=executed > 0.0,
    )
    return executed, vwap, notional


def _directions_to_array(directions: Sequence[str]) -> np.ndarray:
    return np.asarray([1 if direction == "buy" else -1 for direction in directions], dtype=np.int8)


def _make_schedules(
    task_size: int,
    n_steps: int,
    kappa_T_grid: Sequence[float],
    lot_size: int,
) -> Dict[float, np.ndarray]:
    return {
        float(kappa_T): ac_schedule(
            task_size=task_size,
            n_steps=n_steps,
            kappa_T=float(kappa_T),
            lot_size=lot_size,
        ).astype(np.float64)
        for kappa_T in kappa_T_grid
    }


def evaluate_ac_grid_fast(
    snapshots: pd.DataFrame,
    *,
    spec: EpisodeSpec,
    kappa_T_grid: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    depth: Optional[int] = None,
    max_episodes: Optional[int] = None,
    carry_unfilled: bool = True,
    dtype: np.dtype | type = np.float64,
) -> Tuple[pd.DataFrame, None, pd.DataFrame]:
    """Evaluate AC/TWAP schedules using vectorized NumPy replay.

    The tuple shape matches :func:`ac_benchmark.replay.evaluate_ac_grid`.
    Fill-level records are intentionally not produced here; use the slow replay
    path when ``return_fills`` is needed.
    """
    if depth is None:
        depth = infer_depth(snapshots)
    if depth <= 0:
        raise ValueError("Could not infer depth from snapshot columns")

    plan = build_episode_plan(len(snapshots), spec)
    if max_episodes is not None:
        plan = plan.iloc[: int(max_episodes)].copy()
    if plan.empty:
        raise ValueError("No complete episodes could be generated")

    arrays = _depth_arrays(snapshots, depth, dtype=dtype)
    mids = arrays["mid"]
    timestamps = arrays["timestamp"]

    starts = plan["start_row"].to_numpy(dtype=np.int64)
    n_episodes = len(starts)
    n_steps = int(spec.episode_length)
    step_offsets = np.arange(n_steps, dtype=np.int64) * int(spec.step_stride_rows)
    row_indices = starts[:, None] + step_offsets[None, :]
    if row_indices.max() >= len(snapshots):
        raise IndexError("episode index outside snapshot frame")

    directions = _directions_to_array(plan["direction"].tolist())
    buy_positions = np.flatnonzero(directions == 1)
    sell_positions = np.flatnonzero(directions != 1)
    schedules = _make_schedules(spec.task_size, n_steps, kappa_T_grid, spec.lot_size)

    initial_mid = mids[row_indices[:, 0]]
    final_mid = mids[row_indices[:, -1]]
    side_sign = np.where(directions == 1, 1.0, -1.0)
    market_move_signed = side_sign * (final_mid - initial_mid)

    task_size = int(spec.task_size)
    denom = max(1, task_size)
    tick = float(spec.tick_size) if spec.tick_size > 0 else 1.0
    half_remaining = task_size / 2.0

    metric_frames: List[pd.DataFrame] = []
    for kappa_T, schedule in schedules.items():
        remaining = np.full(n_episodes, task_size, dtype=np.float64)
        carry = np.zeros(n_episodes, dtype=np.float64)
        executed_total = np.zeros(n_episodes, dtype=np.float64)
        total_notional = np.zeros(n_episodes, dtype=np.float64)
        slippage = np.zeros(n_episodes, dtype=np.float64)
        sum_sq_remaining = np.zeros(n_episodes, dtype=np.float64)
        max_remaining = np.zeros(n_episodes, dtype=np.float64)
        n_fill_steps = np.zeros(n_episodes, dtype=np.int64)
        completion_step = np.full(n_episodes, n_steps, dtype=np.int64)
        half_completion_step = np.full(n_episodes, n_steps, dtype=np.int64)
        completed = np.zeros(n_episodes, dtype=bool)
        half_completed = np.zeros(n_episodes, dtype=bool)

        for step in range(n_steps):
            if step == n_steps - 1:
                desired = remaining.copy()
            else:
                desired = np.minimum(schedule[step] + (carry if carry_unfilled else 0.0), remaining)

            executed = np.zeros(n_episodes, dtype=np.float64)
            notional = np.zeros(n_episodes, dtype=np.float64)

            if buy_positions.size:
                rows = row_indices[buy_positions, step]
                bought, _, bought_notional = _book_walk_batch(
                    arrays["ask_prices"][rows],
                    arrays["ask_sizes"][rows],
                    desired[buy_positions],
                )
                executed[buy_positions] = bought
                notional[buy_positions] = bought_notional

            if sell_positions.size:
                rows = row_indices[sell_positions, step]
                sold, _, sold_notional = _book_walk_batch(
                    arrays["bid_prices"][rows],
                    arrays["bid_sizes"][rows],
                    desired[sell_positions],
                )
                executed[sell_positions] = sold
                notional[sell_positions] = sold_notional

            executed = np.minimum(executed, remaining)
            remaining -= executed

            if carry_unfilled and step < n_steps - 1:
                carry = np.maximum(0.0, desired - executed)
            else:
                carry.fill(0.0)

            buy_exec = directions == 1
            sell_exec = ~buy_exec
            slippage[buy_exec] += notional[buy_exec] - executed[buy_exec] * initial_mid[buy_exec]
            slippage[sell_exec] += executed[sell_exec] * initial_mid[sell_exec] - notional[sell_exec]

            executed_total += executed
            total_notional += notional
            n_fill_steps += executed > 0.0
            sum_sq_remaining += remaining * remaining
            max_remaining = np.maximum(max_remaining, remaining)

            newly_completed = (~completed) & (remaining <= 0.0)
            completion_step[newly_completed] = step + 1
            completed |= newly_completed

            newly_half_completed = (~half_completed) & (remaining <= half_remaining)
            half_completion_step[newly_half_completed] = step + 1
            half_completed |= newly_half_completed

        episode_vwap = np.divide(
            total_notional,
            executed_total,
            out=np.full(n_episodes, np.nan, dtype=np.float64),
            where=executed_total > 0.0,
        )
        metric_frames.append(
            pd.DataFrame(
                {
                    "episode_id": plan["episode_id"].to_numpy(dtype=np.int64),
                    "start_row": starts,
                    "start_timestamp": timestamps[row_indices[:, 0]],
                    "end_timestamp": timestamps[row_indices[:, -1]],
                    "direction": plan["direction"].to_numpy(),
                    "kappa_T": float(kappa_T),
                    "task_size": task_size,
                    "n_steps": n_steps,
                    "depth": int(depth),
                    "initial_mid": initial_mid,
                    "final_mid": final_mid,
                    "market_move_signed": market_move_signed,
                    "scheduled_total": int(np.sum(schedule)),
                    "executed_quantity": executed_total.astype(np.int64),
                    "unfinished": remaining.astype(np.int64),
                    "completion_rate": executed_total / denom,
                    "episode_vwap": episode_vwap,
                    "slippage_notional": slippage,
                    "slippage_per_share_task": slippage / denom,
                    "slippage_ticks_per_share": slippage / (denom * tick),
                    "avg_sq_remaining": sum_sq_remaining / n_steps,
                    "max_remaining": max_remaining.astype(np.int64),
                    "completion_step": completion_step,
                    "half_completion_step": half_completion_step,
                    "n_fill_steps": n_fill_steps,
                    "aggressive_ratio": 1.0,
                    "carry_unfilled": bool(carry_unfilled),
                }
            )
        )

    return pd.concat(metric_frames, ignore_index=True), None, plan
