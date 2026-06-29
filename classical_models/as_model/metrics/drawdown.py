from __future__ import annotations

import numpy as np


def max_drawdown(values) -> float:
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(series)
    return float(np.nanmin(series - running_max))
