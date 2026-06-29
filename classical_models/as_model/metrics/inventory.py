from __future__ import annotations

import numpy as np


def avg_q2(inventory) -> float:
    q = np.asarray(inventory, dtype=float)
    return float(np.nanmean(q * q)) if q.size else float("nan")


def max_abs_inventory(inventory) -> float:
    q = np.asarray(inventory, dtype=float)
    return float(np.nanmax(np.abs(q))) if q.size else float("nan")
