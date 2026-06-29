from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from script.run_as_baseline import list_input_files, select_split_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ASBaselineCLITest(unittest.TestCase):
    def _copy_lobster_fixture_pair(self, destination: Path, day: str) -> None:
        destination.mkdir(parents=True, exist_ok=True)
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

        (destination / f"AMZN_{day}_34200000_57600000_message_10.csv").write_text(
            "\n".join(message_rows) + "\n",
            encoding="utf-8",
        )
        (destination / f"AMZN_{day}_34200000_57600000_orderbook_10.csv").write_text(
            "\n".join(orderbook_rows) + "\n",
            encoding="utf-8",
        )

    def test_run_as_baseline_tiny_fixture(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            tmp_path = Path(tmp)
            fixture_dir = tmp_path / "lobster_fixture"
            self._copy_lobster_fixture_pair(fixture_dir, "2025-11-12")
            output_dir = tmp_path / "results" / "as"
            baseline_dir = tmp_path / "results" / "baselines"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "script" / "run_as_baseline.py"),
                "--data-format",
                "lobster",
                "--sample-data-dir",
                str(fixture_dir),
                "--output-dir",
                str(output_dir),
                "--baselines-output-dir",
                str(baseline_dir),
                "--no-plots",
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

            expected = [
                output_dir / "as_calibration.json",
                output_dir / "as_gamma_selection.csv",
                output_dir / "as_test_metrics.csv",
                output_dir / "as_test_summary.json",
                output_dir / "as_for_rl_comparison.csv",
                output_dir / "as_run_manifest.json",
                baseline_dir / "no_trade_test_summary.json",
                baseline_dir / "no_trade_for_rl_comparison.csv",
            ]
            for path in expected:
                self.assertTrue(path.exists(), f"missing output: {path}")

            comparison = pd.read_csv(output_dir / "as_for_rl_comparison.csv")
            self.assertFalse(comparison.empty)
            self.assertEqual(set(comparison["strategy"]), {"as_classical_control", "no_trade"})
            self.assertEqual(set(comparison["baseline_family"]), {"classical_control", "passive_control"})
            numeric = comparison.select_dtypes(include=["number"])
            self.assertFalse(numeric.isna().any().any())
            self.assertTrue((comparison[comparison["strategy"].ne("no_trade")]["mean_spread"] > 0).all())
            self.assertTrue((comparison["max_abs_inventory"] <= comparison["inventory_limit"]).all())

            manifest = json.loads((output_dir / "as_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["baseline_family"], "classical_control")
            self.assertEqual(manifest["config"]["horizon_mode"], "finite_episode")

    def test_run_as_baseline_lobster_split_directories(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "lobster_amzn_10"
            self._copy_lobster_fixture_pair(data_dir / "train", "2025-11-12")
            self._copy_lobster_fixture_pair(data_dir / "val", "2026-03-03")
            self._copy_lobster_fixture_pair(data_dir / "test", "2026-04-08")
            output_dir = tmp_path / "results" / "AS"
            baseline_dir = tmp_path / "results" / "baselines"

            command = [
                sys.executable,
                str(PROJECT_ROOT / "script" / "run_as_baseline.py"),
                "--data-format",
                "lobster",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--baselines-output-dir",
                str(baseline_dir),
                "--no-plots",
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

            calibration = json.loads((output_dir / "as_calibration.json").read_text(encoding="utf-8"))
            self.assertEqual(calibration["train_start"], "2025-11-12")
            self.assertEqual(calibration["validation_start"], "2026-03-03")
            self.assertEqual(calibration["test_start"], "2026-04-08")
            self.assertEqual(calibration["fill_model"], "lobster_level")

            comparison = pd.read_csv(output_dir / "as_for_rl_comparison.csv")
            self.assertEqual(set(comparison["date"]), {"2026-04-08"})
            self.assertEqual(set(comparison["strategy"]), {"as_classical_control", "no_trade"})
            self.assertFalse(comparison.select_dtypes(include=["number"]).isna().any().any())

            summary = json.loads((output_dir / "as_test_summary.json").read_text(encoding="utf-8"))
            self.assertIn("as_classical_control", summary["strategies"])
            self.assertIn("no_trade", summary["strategies"])

    def test_run_as_baseline_lobster_tiny_fixture(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            tmp_path = Path(tmp)
            fixture_dir = tmp_path / "lobster_fixture"
            self._copy_lobster_fixture_pair(fixture_dir, "2025-11-12")
            output_dir = tmp_path / "results" / "AS"
            baseline_dir = tmp_path / "results" / "baselines"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "script" / "run_as_baseline.py"),
                "--data-format",
                "lobster",
                "--sample-data-dir",
                str(fixture_dir),
                "--output-dir",
                str(output_dir),
                "--baselines-output-dir",
                str(baseline_dir),
                "--no-plots",
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

            expected = [
                output_dir / "as_calibration.json",
                output_dir / "as_gamma_selection.csv",
                output_dir / "as_test_metrics.csv",
                output_dir / "as_test_summary.json",
                output_dir / "as_for_rl_comparison.csv",
                output_dir / "as_run_manifest.json",
                baseline_dir / "no_trade_test_summary.json",
                baseline_dir / "no_trade_for_rl_comparison.csv",
            ]
            for path in expected:
                self.assertTrue(path.exists(), f"missing output: {path}")

            calibration = json.loads((output_dir / "as_calibration.json").read_text(encoding="utf-8"))
            self.assertEqual(calibration["data_format"], "lobster")
            self.assertEqual(calibration["fill_model"], "lobster_level")
            self.assertEqual(calibration["lobster_levels"], 10)

            gamma_selection = pd.read_csv(output_dir / "as_gamma_selection.csv")
            self.assertEqual(int(gamma_selection["selected"].sum()), 1)

            comparison = pd.read_csv(output_dir / "as_for_rl_comparison.csv")
            self.assertFalse(comparison.empty)
            self.assertEqual(set(comparison["strategy"]), {"as_classical_control", "no_trade"})
            numeric = comparison.select_dtypes(include=["number"])
            self.assertFalse(numeric.isna().any().any())
            self.assertTrue((comparison[comparison["strategy"].ne("no_trade")]["mean_spread"] > 0).all())
            self.assertTrue((comparison["max_abs_inventory"] <= comparison["inventory_limit"]).all())

    def test_resume_from_calibration_matches_fresh_fixture_run(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            tmp_path = Path(tmp)
            fixture_dir = tmp_path / "lobster_fixture"
            self._copy_lobster_fixture_pair(fixture_dir, "2025-11-12")
            fresh_dir = tmp_path / "fresh"
            resumed_dir = tmp_path / "resumed"

            base_command = [
                sys.executable,
                str(PROJECT_ROOT / "script" / "run_as_baseline.py"),
                "--data-format",
                "lobster",
                "--sample-data-dir",
                str(fixture_dir),
                "--no-plots",
            ]
            subprocess.run(
                base_command
                + [
                    "--output-dir",
                    str(fresh_dir),
                    "--baselines-output-dir",
                    str(fresh_dir / "baselines"),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                base_command
                + [
                    "--output-dir",
                    str(resumed_dir),
                    "--baselines-output-dir",
                    str(resumed_dir / "baselines"),
                    "--resume-from-calibration",
                    str(fresh_dir / "as_calibration.json"),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            fresh_gamma = pd.read_csv(fresh_dir / "as_gamma_selection.csv")
            resumed_gamma = pd.read_csv(resumed_dir / "as_gamma_selection.csv")
            self.assertEqual(
                float(fresh_gamma.loc[fresh_gamma["selected"], "gamma"].iloc[0]),
                float(resumed_gamma.loc[resumed_gamma["selected"], "gamma"].iloc[0]),
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(fresh_dir / "as_test_metrics.csv"),
                pd.read_csv(resumed_dir / "as_test_metrics.csv"),
            )

            manifest = json.loads((resumed_dir / "as_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["calibration_source"], "resumed")

    def test_include_symmetric_adds_auxiliary_strategy(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            tmp_path = Path(tmp)
            fixture_dir = tmp_path / "lobster_fixture"
            self._copy_lobster_fixture_pair(fixture_dir, "2025-11-12")
            output_dir = tmp_path / "results" / "AS"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "script" / "run_as_baseline.py"),
                "--data-format",
                "lobster",
                "--sample-data-dir",
                str(fixture_dir),
                "--output-dir",
                str(output_dir),
                "--baselines-output-dir",
                str(tmp_path / "results" / "baselines"),
                "--include-symmetric",
                "--no-plots",
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
            comparison = pd.read_csv(output_dir / "as_for_rl_comparison.csv")
            self.assertEqual(set(comparison["strategy"]), {"as_classical_control", "symmetric_mm", "no_trade"})

    def test_split_selection_prefers_named_split_directories(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            data_dir = Path(tmp) / "AMZN"
            for split, dates in {
                "train": ["20250102", "20250103"],
                "validation": ["20250106"],
                "test": ["20250107"],
            }.items():
                split_dir = data_dir / split
                split_dir.mkdir(parents=True)
                for date in dates:
                    (split_dir / f"xnas-itch-{date}.mbo.dbn.zst").write_text("", encoding="utf-8")

            all_files = list_input_files(data_dir)
            selected_train = select_split_files(
                data_dir=data_dir,
                files=all_files,
                sample_data=None,
                start=None,
                end=None,
                start_index=1,
                end_index=99,
                split_name="train",
            )
            self.assertEqual([path.parent.name for path in selected_train], ["train", "train"])
            self.assertEqual([path.name for path in selected_train], [
                "xnas-itch-20250102.mbo.dbn.zst",
                "xnas-itch-20250103.mbo.dbn.zst",
            ])


if __name__ == "__main__":
    unittest.main()
