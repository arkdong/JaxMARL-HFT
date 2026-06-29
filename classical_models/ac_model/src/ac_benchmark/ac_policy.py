from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _largest_remainder_integer_allocation(raw: np.ndarray, total: int) -> np.ndarray:
    """Convert non-negative raw sizes to non-negative integers summing to total."""
    if total < 0:
        raise ValueError("total must be non-negative")
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1:
        raise ValueError("raw must be one-dimensional")
    if len(raw) == 0:
        raise ValueError("raw must not be empty")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw contains non-finite values")
    raw = np.maximum(raw, 0.0)
    raw_sum = raw.sum()
    if raw_sum <= 0:
        out = np.zeros_like(raw, dtype=int)
        out[-1] = total
        return out

    scaled = raw * (total / raw_sum)
    floors = np.floor(scaled).astype(int)
    remainder = int(total - floors.sum())
    if remainder > 0:
        frac_order = np.argsort(-(scaled - floors), kind="stable")
        floors[frac_order[:remainder]] += 1
    elif remainder < 0:
        # Rare numerical case: remove from the smallest fractional parts with positive lots.
        frac_order = np.argsort(scaled - floors, kind="stable")
        need = -remainder
        for idx in frac_order:
            take = min(need, floors[idx])
            floors[idx] -= take
            need -= take
            if need == 0:
                break
        if need:
            raise RuntimeError("integer allocation failed")
    return floors


def ac_holdings(task_size: int, n_steps: int, kappa_T: float) -> np.ndarray:
    """Return continuous AC holdings x_0,...,x_N.

    Parameters
    ----------
    task_size:
        Initial absolute execution quantity X.
    n_steps:
        Number of decision intervals N.
    kappa_T:
        Dimensionless risk-aversion / front-loading parameter. kappa_T=0 is the
        TWAP limit. Larger values front-load execution.

    Notes
    -----
    We normalise the episode horizon to T=1. The discrete AC schedule is

        x_j = X sinh(kappa (T - t_j)) / sinh(kappa T)

    with kappa_T = kappa*T. Since T=1 in normalised time, kappa=kappa_T.
    """
    if task_size < 0:
        raise ValueError("task_size must be non-negative")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if kappa_T < 0:
        raise ValueError("kappa_T must be non-negative")

    X = float(task_size)
    N = int(n_steps)
    if abs(kappa_T) < 1e-12:
        return np.linspace(X, 0.0, N + 1)

    j = np.arange(N + 1, dtype=float)
    u = j / float(N)
    denom = math.sinh(float(kappa_T))
    holdings = X * np.sinh(float(kappa_T) * (1.0 - u)) / denom
    holdings[0] = X
    holdings[-1] = 0.0
    return holdings


def ac_schedule(
    task_size: int,
    n_steps: int,
    kappa_T: float,
    lot_size: int = 1,
) -> np.ndarray:
    """Return integer trade sizes n_1,...,n_N summing to task_size.

    The allocation follows the continuous AC trajectory and then uses a largest
    remainder rule so that the integer schedule exactly sums to the task size.
    For lot_size>1, it allocates complete lots first and places any residual
    shares in the final step.
    """
    if task_size < 0:
        raise ValueError("task_size must be non-negative")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    if task_size == 0:
        return np.zeros(n_steps, dtype=int)

    holdings = ac_holdings(task_size=task_size, n_steps=n_steps, kappa_T=kappa_T)
    raw = np.maximum(holdings[:-1] - holdings[1:], 0.0)

    if lot_size == 1:
        return _largest_remainder_integer_allocation(raw, task_size)

    full_lots = task_size // lot_size
    residual = task_size - full_lots * lot_size
    lot_schedule = _largest_remainder_integer_allocation(raw / float(lot_size), full_lots)
    shares = lot_schedule.astype(int) * int(lot_size)
    # Put residual shares at the final decision to preserve exact completion.
    shares[-1] += residual
    assert int(shares.sum()) == int(task_size)
    return shares


def schedule_summary(task_size: int, n_steps: int, kappa_T: float, lot_size: int = 1) -> dict:
    """Small diagnostic summary for logging and tables."""
    sched = ac_schedule(task_size, n_steps, kappa_T, lot_size)
    rem = task_size - np.cumsum(sched)
    rem_path = np.concatenate([[task_size], rem])
    half_idx = int(np.argmax(rem_path <= task_size / 2.0)) if task_size > 0 else 0
    if rem_path[half_idx] > task_size / 2.0:
        half_idx = n_steps
    return {
        "task_size": int(task_size),
        "n_steps": int(n_steps),
        "kappa_T": float(kappa_T),
        "lot_size": int(lot_size),
        "first_trade": int(sched[0]),
        "last_trade": int(sched[-1]),
        "max_trade": int(sched.max(initial=0)),
        "sum_trades": int(sched.sum()),
        "avg_sq_remaining": float(np.mean(np.square(rem))),
        "half_completion_step": int(half_idx),
    }

