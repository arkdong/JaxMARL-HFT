from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from as_model import ASParams, ASState, compute_as_quote  # noqa: E402


class PureASModelTest(unittest.TestCase):
    def test_positive_gamma_quote_formula_and_finite_horizon_tau(self) -> None:
        params = ASParams(
            sigma=0.02,
            k=30.0,
            gamma=0.1,
            horizon_seconds=10.0,
            tick_size=0.01,
        )
        quote = compute_as_quote(
            params,
            ASState(mid=100.0, inventory=5, elapsed_seconds=4.0),
            horizon_mode="finite_episode",
        )

        tau = 6.0
        expected_reservation = 100.0 - 5 * params.gamma * params.sigma * params.sigma * tau
        expected_raw_spread = (
            params.gamma * params.sigma * params.sigma * tau
            + (2.0 / params.gamma) * math.log1p(params.gamma / params.k)
        )

        self.assertAlmostEqual(quote.reservation_price, expected_reservation)
        self.assertAlmostEqual(quote.raw_ask - quote.raw_bid, expected_raw_spread)
        self.assertEqual(round((quote.ask or 0.0) / params.tick_size), (quote.ask or 0.0) / params.tick_size)
        self.assertEqual(round((quote.bid or 0.0) / params.tick_size), (quote.bid or 0.0) / params.tick_size)
        self.assertGreaterEqual(quote.spread, params.tick_size)

    def test_zero_gamma_uses_risk_neutral_spread_limit(self) -> None:
        params = ASParams(
            sigma=0.02,
            k=50.0,
            gamma=0.0,
            horizon_seconds=10.0,
            tick_size=0.01,
        )
        quote = compute_as_quote(
            params,
            ASState(mid=100.0, inventory=10, elapsed_seconds=0.0),
            horizon_mode="rolling",
        )
        self.assertEqual(quote.reservation_price, 100.0)
        self.assertAlmostEqual(quote.raw_ask - quote.raw_bid, 2.0 / params.k)

    def test_inventory_limit_suppresses_one_side(self) -> None:
        params = ASParams(
            sigma=0.02,
            k=50.0,
            gamma=0.1,
            horizon_seconds=10.0,
            tick_size=0.01,
        )
        quote = compute_as_quote(
            params,
            ASState(mid=100.0, inventory=10, elapsed_seconds=0.0),
            inventory_limit=10,
            order_size=1,
        )
        self.assertIsNone(quote.bid)
        self.assertIsNotNone(quote.ask)


if __name__ == "__main__":
    unittest.main()
