from __future__ import annotations

import numpy as np


def fill_rate(num_fills: float, num_orders: float) -> float:
    return float(num_fills / num_orders) if num_orders > 0 else 0.0


def order_submission_rate(num_orders: float, num_steps: int, *, sides_per_step: int = 2) -> float:
    denom = num_steps * sides_per_step
    return float(num_orders / denom) if denom > 0 else 0.0


def no_trade_rate(fills_per_step) -> float:
    fills = np.asarray(fills_per_step, dtype=float)
    return float(np.mean(fills <= 0)) if fills.size else 1.0
