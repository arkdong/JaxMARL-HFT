from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ac_policy import ac_schedule
from .calibration import book_walk_vwap
from .data import build_episode_plan, episode_indices, infer_depth
from .schema import Direction, EpisodeSpec


def evaluate_ac_episode(
    snapshots: pd.DataFrame,
    indices: Sequence[int],
    *,
    direction: Direction,
    task_size: int,
    kappa_T: float,
    lot_size: int = 10,
    tick_size: float = 0.01,
    depth: Optional[int] = None,
    episode_id: Optional[int] = None,
    start_row: Optional[int] = None,
    carry_unfilled: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Replay one AC execution episode using marketable book-walking orders.

    The benchmark is intentionally deterministic: the AC policy determines a
    quantity schedule, and each slice consumes displayed depth on the opposite
    side of the historical book snapshot.
    """
    if depth is None:
        depth = infer_depth(snapshots)
    if depth <= 0:
        raise ValueError("snapshots must contain at least one complete bid/ask level")

    indices = np.asarray(indices, dtype=int)
    if len(indices) <= 0:
        raise ValueError("indices must not be empty")
    if indices.min() < 0 or indices.max() >= len(snapshots):
        raise IndexError("episode indices out of snapshot range")

    n_steps = len(indices)
    schedule = ac_schedule(task_size=task_size, n_steps=n_steps, kappa_T=kappa_T, lot_size=lot_size)
    p0 = float(snapshots.iloc[indices[0]]["mid_price"])
    final_mid = float(snapshots.iloc[indices[-1]]["mid_price"])
    timestamps = snapshots.iloc[indices]["timestamp"].to_numpy()

    remaining = int(task_size)
    carry = 0
    fill_records: List[Dict[str, float]] = []
    remaining_path: List[int] = []
    scheduled_path: List[int] = []
    executed_path: List[int] = []

    for step, row_idx in enumerate(indices):
        row = snapshots.iloc[int(row_idx)]
        scheduled = int(schedule[step])
        desired = scheduled + (carry if carry_unfilled else 0)
        desired = min(desired, remaining)
        if step == n_steps - 1:
            desired = remaining

        if desired > 0:
            executed, vwap, notional = book_walk_vwap(row, direction, desired, depth)
        else:
            executed, vwap, notional = 0, float("nan"), 0.0

        executed = int(min(executed, remaining))
        remaining -= executed
        if carry_unfilled and step < n_steps - 1:
            carry = max(0, desired - executed)
        else:
            carry = 0

        scheduled_path.append(scheduled)
        executed_path.append(executed)
        remaining_path.append(int(remaining))

        if executed > 0:
            fill_records.append(
                {
                    "episode_id": -1 if episode_id is None else int(episode_id),
                    "step": int(step),
                    "row_idx": int(row_idx),
                    "timestamp": row["timestamp"],
                    "direction": direction,
                    "kappa_T": float(kappa_T),
                    "scheduled_qty": int(scheduled),
                    "executed_qty": int(executed),
                    "vwap": float(vwap),
                    "notional": float(notional),
                    "mid_price": float(row["mid_price"]),
                    "remaining": int(remaining),
                }
            )

    fills = pd.DataFrame(fill_records)
    executed_total = int(np.sum(executed_path))
    if executed_total > 0:
        total_notional = float(fills["notional"].sum()) if not fills.empty else 0.0
        episode_vwap = total_notional / executed_total
    else:
        total_notional = 0.0
        episode_vwap = float("nan")

    if not fills.empty:
        if direction == "buy":
            slippage_notional = float(np.sum(fills["executed_qty"] * (fills["vwap"] - p0)))
        else:
            slippage_notional = float(np.sum(fills["executed_qty"] * (p0 - fills["vwap"])))
    else:
        slippage_notional = 0.0

    rem_arr = np.asarray(remaining_path, dtype=float)
    completion_step = next((i + 1 for i, x in enumerate(remaining_path) if x == 0), n_steps)
    half_remaining = task_size / 2.0
    half_completion_step = next((i + 1 for i, x in enumerate(remaining_path) if x <= half_remaining), n_steps)
    denom = max(1, int(task_size))
    tick = float(tick_size) if tick_size > 0 else 1.0
    side_sign = 1.0 if direction == "buy" else -1.0
    market_move_signed = side_sign * (final_mid - p0)

    metrics: Dict[str, float] = {
        "episode_id": -1 if episode_id is None else int(episode_id),
        "start_row": int(indices[0] if start_row is None else start_row),
        "start_timestamp": timestamps[0],
        "end_timestamp": timestamps[-1],
        "direction": direction,
        "kappa_T": float(kappa_T),
        "task_size": int(task_size),
        "n_steps": int(n_steps),
        "depth": int(depth),
        "initial_mid": float(p0),
        "final_mid": float(final_mid),
        "market_move_signed": float(market_move_signed),
        "scheduled_total": int(np.sum(schedule)),
        "executed_quantity": int(executed_total),
        "unfinished": int(remaining),
        "completion_rate": float(executed_total / denom),
        "episode_vwap": float(episode_vwap),
        "slippage_notional": float(slippage_notional),
        "slippage_per_share_task": float(slippage_notional / denom),
        "slippage_ticks_per_share": float(slippage_notional / (denom * tick)),
        "avg_sq_remaining": float(np.mean(rem_arr * rem_arr)) if len(rem_arr) else float("nan"),
        "max_remaining": int(np.max(rem_arr)) if len(rem_arr) else int(task_size),
        "completion_step": int(completion_step),
        "half_completion_step": int(half_completion_step),
        "n_fill_steps": int(np.sum(np.asarray(executed_path) > 0)),
        "aggressive_ratio": 1.0,
        "carry_unfilled": bool(carry_unfilled),
    }
    return metrics, fills


def evaluate_ac_grid(
    snapshots: pd.DataFrame,
    *,
    spec: EpisodeSpec,
    kappa_T_grid: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0),
    depth: Optional[int] = None,
    max_episodes: Optional[int] = None,
    return_fills: bool = False,
    carry_unfilled: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Evaluate AC/TWAP schedules on every complete episode and kappa_T value."""
    if depth is None:
        depth = infer_depth(snapshots)
    if depth <= 0:
        raise ValueError("Could not infer book depth from snapshots")

    plan = build_episode_plan(len(snapshots), spec)
    if max_episodes is not None:
        plan = plan.iloc[: int(max_episodes)].copy()
    if plan.empty:
        raise ValueError(
            "No complete episodes could be generated. Check episode_length, messages_per_step, and input size."
        )

    metric_records: List[Dict[str, float]] = []
    fill_frames: List[pd.DataFrame] = []
    for _, ep in plan.iterrows():
        idx = episode_indices(int(ep["start_row"]), spec)
        direction = ep["direction"]
        for kappa_T in kappa_T_grid:
            metrics, fills = evaluate_ac_episode(
                snapshots,
                idx,
                direction=direction,
                task_size=spec.task_size,
                kappa_T=float(kappa_T),
                lot_size=spec.lot_size,
                tick_size=spec.tick_size,
                depth=depth,
                episode_id=int(ep["episode_id"]),
                start_row=int(ep["start_row"]),
                carry_unfilled=carry_unfilled,
            )
            metric_records.append(metrics)
            if return_fills and not fills.empty:
                fill_frames.append(fills)

    metrics_df = pd.DataFrame.from_records(metric_records)
    fills_df = pd.concat(fill_frames, ignore_index=True) if (return_fills and fill_frames) else None
    return metrics_df, fills_df, plan

