from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from avellaneda_stoikov_amzn import ASParams
from lobster_as import (
    calibrate_lobster_arrival_rates,
    discover_lobster_pairs,
    is_lobster_dir,
    load_lobster_day,
    simulate_lobster_replay,
)
from script.run_as_baseline import select_lobster_split, simulate_episode_replay


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LobsterASTest(unittest.TestCase):
    def _write_empty_pair(self, root: Path, day: str) -> None:
        for kind in ("message", "orderbook"):
            (root / f"AMZN_{day}_34200000_57600000_{kind}_10.csv").write_text("", encoding="utf-8")

    def _write_lobster_fixture_pair(self, root: Path, day: str = "2025-11-12") -> None:
        root.mkdir(parents=True, exist_ok=True)
        message_rows: list[str] = []
        orderbook_rows: list[str] = []
        start = 34_200.0
        for idx in range(24):
            timestamp = start + idx * 0.25
            event_type = 1
            size = 10
            price = 1_000_100
            direction = 1
            if idx == 1:
                event_type, size, price, direction = 4, 20, 1_000_200, -1
            elif idx == 3:
                event_type, size, price, direction = 4, 20, 1_000_200, 1
            elif idx == 5:
                event_type, size, price, direction = 4, 20, 1_001_000, -1
            elif idx == 7:
                event_type, size, price, direction = 4, 20, 999_000, 1
            message_rows.append(f"{timestamp:.6f},{event_type},{idx + 1},{size},{price},{direction}")

            shift = (idx % 5) * 100
            ask_1 = 1_000_100 + shift
            bid_1 = 999_900 + shift
            levels: list[str] = []
            for level in range(10):
                levels.extend([str(ask_1 + level * 100), "5", str(bid_1 - level * 100), "5"])
            orderbook_rows.append(",".join(levels))

        (root / f"AMZN_{day}_34200000_57600000_message_10.csv").write_text(
            "\n".join(message_rows) + "\n",
            encoding="utf-8",
        )
        (root / f"AMZN_{day}_34200000_57600000_orderbook_10.csv").write_text(
            "\n".join(orderbook_rows) + "\n",
            encoding="utf-8",
        )

    def test_pair_discovery_date_parsing_and_chronological_split_counts(self) -> None:
        holidays = {
            "2025-11-27",
            "2025-12-25",
            "2026-01-01",
            "2026-01-19",
            "2026-02-16",
            "2026-04-03",
        }

        def trading_dates(start: str, end: str) -> list[str]:
            return [
                str(day.date())
                for day in pd.bdate_range(start, end)
                if str(day.date()) not in holidays
            ]

        train_dates = trading_dates("2025-11-12", "2026-03-02")
        validation_dates = trading_dates("2026-03-03", "2026-04-07")
        test_dates = trading_dates("2026-04-08", "2026-05-11")
        self.assertEqual(len(train_dates), 74)
        self.assertEqual(len(validation_dates), 25)
        self.assertEqual(len(test_dates), 24)

        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            root = Path(tmp)
            for day in train_dates + validation_dates + test_dates:
                for kind in ("message", "orderbook"):
                    (root / f"AMZN_{day}_34200000_57600000_{kind}_10.csv").write_text("", encoding="utf-8")

            pairs = discover_lobster_pairs(root)
            self.assertEqual(len(pairs), 123)
            self.assertEqual(pairs[0].date.isoformat(), "2025-11-12")
            self.assertEqual(pairs[-1].date.isoformat(), "2026-05-11")
            self.assertEqual(pairs[0].start_seconds, 34200.0)
            self.assertEqual(pairs[0].end_seconds, 57600.0)

            self.assertEqual(len(select_lobster_split(pairs=pairs, sample_data_dir=None, start=None, end=None, split_name="train")), 74)
            self.assertEqual(
                len(select_lobster_split(pairs=pairs, sample_data_dir=None, start=None, end=None, split_name="validation")),
                25,
            )
            self.assertEqual(len(select_lobster_split(pairs=pairs, sample_data_dir=None, start=None, end=None, split_name="test")), 24)

    def test_split_directories_with_val_alias_are_selected_directly(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            root = Path(tmp)
            split_dates = {
                "train": ["2025-11-12", "2025-11-13"],
                "val": ["2026-03-03"],
                "test": ["2026-04-08"],
            }
            for dirname, dates in split_dates.items():
                split_dir = root / dirname
                split_dir.mkdir()
                for day in dates:
                    self._write_empty_pair(split_dir, day)

            self.assertTrue(is_lobster_dir(root))
            train = select_lobster_split(data_dir=root, sample_data_dir=None, start=None, end=None, split_name="train")
            validation = select_lobster_split(
                data_dir=root,
                sample_data_dir=None,
                start=None,
                end=None,
                split_name="validation",
            )
            test = select_lobster_split(data_dir=root, sample_data_dir=None, start=None, end=None, split_name="test")

            self.assertEqual([pair.date.isoformat() for pair in train], ["2025-11-12", "2025-11-13"])
            self.assertEqual([pair.message_path.parent.name for pair in train], ["train", "train"])
            self.assertEqual([pair.date.isoformat() for pair in validation], ["2026-03-03"])
            self.assertEqual(validation[0].message_path.parent.name, "val")
            self.assertEqual([pair.date.isoformat() for pair in test], ["2026-04-08"])

    def test_bbo_sampling_and_price_scaling(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            root = Path(tmp)
            self._write_lobster_fixture_pair(root)
            pair = discover_lobster_pairs(root)[0]
            day = load_lobster_day(pair, dt_seconds=0.25)
            self.assertEqual(len(day.bbo), 24)
            self.assertAlmostEqual(float(day.bbo["best_ask"].iloc[0]), 100.01)
            self.assertAlmostEqual(float(day.bbo["best_bid"].iloc[0]), 99.99)
            self.assertAlmostEqual(float(day.bbo["spread"].iloc[0]), 0.02)
            self.assertEqual(int(day.bbo["ask_price_int_1"].iloc[0]), 1000100)
            self.assertIsNotNone(day.bbo.index.tz)

    def test_visible_execution_direction_mapping(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            root = Path(tmp)
            self._write_lobster_fixture_pair(root)
            pair = discover_lobster_pairs(root)[0]
            day = load_lobster_day(pair, dt_seconds=0.25)
            self.assertEqual(day.trade_flow.buy_reaching_ask(1, 1000200), 20)
            self.assertEqual(day.trade_flow.buy_reaching_ask(1, 1000300), 0)
            self.assertEqual(day.trade_flow.sell_reaching_bid(3, 1000200), 20)
            self.assertEqual(day.trade_flow.sell_reaching_bid(3, 1000100), 0)

    def test_lobster_level_fill_accounting_and_inventory_limit(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            root = Path(tmp)
            self._write_lobster_fixture_pair(root)
            pair = discover_lobster_pairs(root)[0]
            day = load_lobster_day(pair, dt_seconds=0.25)
            params = ASParams(
                gamma=0.0,
                sigma=0.01,
                k=100.0,
                A=1.0,
                horizon_seconds=6.0,
                dt_seconds=0.25,
                tick_size=0.01,
                order_size=10,
                q_max=10,
            )
            replay = simulate_lobster_replay(day.bbo, day.trade_flow, params, strategy="inventory")
            self.assertFalse(replay.empty)
            self.assertGreater(int(replay["bid_fill"].sum() + replay["ask_fill"].sum()), 0)
            self.assertTrue((replay["inventory"].abs() <= params.q_max).all())
            self.assertIn(5, set(replay["ask_queue_ahead"].astype(int)))

    def test_lobster_level_calibration_positive_arrival_fit(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            root = Path(tmp)
            self._write_lobster_fixture_pair(root)
            pair = discover_lobster_pairs(root)[0]
            day = load_lobster_day(pair, dt_seconds=0.25)
            calibration = calibrate_lobster_arrival_rates(
                day.bbo,
                day.trade_flow,
                deltas=np.arange(1, 5) * 0.01,
                dt_seconds=0.25,
                tick_size=0.01,
                order_size=10,
            )
            self.assertGreater(calibration.A, 0)
            self.assertGreater(calibration.k, 0)
            self.assertFalse(calibration.to_frame().empty)

    def test_episode_replay_resets_local_state_and_elapsed_time(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            root = Path(tmp)
            self._write_lobster_fixture_pair(root)
            pair = discover_lobster_pairs(root)[0]
            day = load_lobster_day(pair, dt_seconds=0.25)
            params = ASParams(
                gamma=0.05,
                sigma=0.01,
                k=100.0,
                A=1.0,
                horizon_seconds=1.5,
                dt_seconds=0.25,
                tick_size=0.01,
                order_size=10,
                q_max=20,
                rolling_horizon=False,
            )
            args = argparse.Namespace(fill_model="lobster_level")

            first = simulate_episode_replay(day, start=0, end=6, params=params, args=args, replay_strategy="inventory")
            second = simulate_episode_replay(day, start=6, end=12, params=params, args=args, replay_strategy="inventory")

            self.assertFalse(first.empty)
            self.assertFalse(second.empty)
            self.assertAlmostEqual(float(first["tau"].iloc[0]), params.horizon_seconds)
            self.assertAlmostEqual(float(second["tau"].iloc[0]), params.horizon_seconds)
            self.assertEqual(int(first["inventory"].iloc[0]), 0)
            self.assertEqual(int(second["inventory"].iloc[0]), 0)
            self.assertAlmostEqual(float(first["cash"].iloc[0]), 0.0)
            self.assertAlmostEqual(float(second["cash"].iloc[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
