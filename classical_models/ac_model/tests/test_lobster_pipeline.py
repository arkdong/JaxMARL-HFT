from __future__ import annotations

from pathlib import Path

import pandas as pd

from ac_benchmark.cli import main
from ac_benchmark.lobster_pipeline import select_lobster_split


def _write_lobster_day(root: Path, split: str, day: str, *, rows: int = 180, levels: int = 2) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    base = f"AMZN_{day}_34200000_57600000"
    message_path = split_dir / f"{base}_message_{levels}.csv"
    orderbook_path = split_dir / f"{base}_orderbook_{levels}.csv"

    messages = []
    books = []
    for idx in range(rows):
        mid = 100.0 + 0.001 * idx
        messages.append([34200.0 + idx * 0.1, 1, idx + 1, 10, int(mid * 10000), 1])
        book_row = []
        for level in range(1, levels + 1):
            offset = 0.01 * level
            book_row.extend(
                [
                    int((mid + offset) * 10000),
                    1000,
                    int((mid - offset) * 10000),
                    1000,
                ]
            )
        books.append(book_row)

    pd.DataFrame(messages).to_csv(message_path, index=False, header=False)
    pd.DataFrame(books).to_csv(orderbook_path, index=False, header=False)


def test_select_lobster_split_directories(tmp_path: Path) -> None:
    root = tmp_path / "lobster_amzn_10"
    _write_lobster_day(root, "train", "2025-11-12")
    _write_lobster_day(root, "validation", "2026-03-03")
    _write_lobster_day(root, "test", "2026-04-08")

    train = select_lobster_split(data_dir=root, levels=2, split_name="train")
    validation = select_lobster_split(data_dir=root, levels=2, split_name="validation")
    test = select_lobster_split(data_dir=root, levels=2, split_name="test")

    assert [item.date.isoformat() for item in train] == ["2025-11-12"]
    assert [item.date.isoformat() for item in validation] == ["2026-03-03"]
    assert [item.date.isoformat() for item in test] == ["2026-04-08"]


def test_run_lobster_pipeline_tiny_fixture(tmp_path: Path) -> None:
    root = tmp_path / "lobster_amzn_10"
    _write_lobster_day(root, "train", "2025-11-12")
    _write_lobster_day(root, "validation", "2026-03-03")
    _write_lobster_day(root, "test", "2026-04-08")
    out_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"

    main(
        [
            "run-lobster",
            "--data-format",
            "lobster",
            "--data-dir",
            str(root),
            "--output-dir",
            str(out_dir),
            "--cache-dir",
            str(cache_dir),
            "--lobster-levels",
            "2",
            "--q-grid",
            "10,20",
            "--kappa-grid",
            "0,1",
            "--task-size",
            "20",
            "--episode-length",
            "8",
            "--messages-per-step",
            "1",
            "--episode-start-frequency-steps",
            "8",
            "--directions",
            "buy",
            "--max-episodes",
            "2",
            "--n-boot",
            "0",
            "--no-plots",
        ]
    )

    expected = [
        "ac_calibration.json",
        "ac_daily_calibration.csv",
        "ac_kappa_selection.csv",
        "ac_test_metrics.csv",
        "ac_for_rl_comparison.csv",
        "ac_run_manifest.json",
    ]
    for name in expected:
        assert (out_dir / name).exists()

    selection = pd.read_csv(out_dir / "ac_kappa_selection.csv")
    assert selection["selected"].sum() == 1

    comparison = pd.read_csv(out_dir / "ac_for_rl_comparison.csv")
    assert not comparison.empty
    assert set(comparison["split"]) == {"test"}
    assert comparison["episode_id"].str.startswith("2026-04-08_").all()
