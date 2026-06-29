from __future__ import annotations

import numpy as np


def portfolio_value(cash, inventory, mid_price):
    """Marked-to-market value: cash plus inventory valued at the reference mid."""
    return np.asarray(cash, dtype=float) + np.asarray(inventory, dtype=float) * np.asarray(mid_price, dtype=float)
