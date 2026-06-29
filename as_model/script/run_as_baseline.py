#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence, Union

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from avellaneda_stoikov_amzn import (  # noqa: E402
    ASParams,
    ArrivalCalibration,
    build_interval_trade_flow,
    build_sampled_bbo,
    calibrate_arrival_rates,
    calibrate_queue_arrival_rates,
    choose_gamma_for_target_skew,
    estimate_sigma_from_mid,
    filter_regular_hours,
    interval_trade_extrema,
    load_mbo,
    simulate_replay,
)
from lobster_as import (  # noqa: E402
    DEFAULT_LOBSTER_LEVELS,
    LobsterDayFiles,
    LobsterTradeFlow,
    calibrate_lobster_arrival_rates,
    discover_lobster_pairs,
    is_lobster_dir,
    load_lobster_day,
    simulate_lobster_replay,
)
from metrics import (  # noqa: E402
    avg_q2,
    fill_rate,
    max_abs_inventory,
    max_drawdown,
    no_trade_rate,
    order_submission_rate,
    portfolio_value,
)


DATE_RE = re.compile(r"(\d{8})")
ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
LOBSTER_SPLIT_DATES = {
    "train": ("2025-11-12", "2026-03-02"),
    "validation": ("2026-03-03", "2026-04-07"),
    "test": ("2026-04-08", "2026-05-11"),
}
LOBSTER_SPLIT_DIR_ALIASES = {
    "train": ("train",),
    "validation": ("validation", "val"),
    "test": ("test",),
}
DEBUG_FIGURES = [
    "mid_price_path.png",
    "fill_probability_curve.png",
    "log_arrival_fit.png",
    "daily_volatility_distribution.png",
    "single_day_quotes.png",
    "single_day_inventory.png",
    "single_day_pnl.png",
    "inventory_skew_check.png",
    "gamma_sweep_full.png",
]
PLACEHOLDER_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfeA\xe2~\x8b\x00\x00\x00\x00IEND\xaeB`\x82"
)


@dataclass(frozen=True)
class CalibrationResult:
    sigma: float
    A: float
    k: float
    data_format: str
    lobster_levels: int | None
    num_train_days: int
    sigma_method: str
    arrival_fit_method: str
    decision_interval_seconds: float
    fill_model: str
    order_size: int
    inventory_limit: int
    tick_size: float
    horizon_mode: str
    horizon_seconds: float
    gamma_seed: float
    train_start: str | None
    train_end: str | None
    validation_start: str | None
    validation_end: str | None
    test_start: str | None
    test_end: str | None
    daily_calibrations: list[dict[str, float | str]]


@dataclass(frozen=True)
class DayData:
    date: str
    mbo: pd.DataFrame | None
    mbo_rth: pd.DataFrame | None
    bbo: pd.DataFrame
    extrema: pd.DataFrame
    trade_flow: Any
    data_format: str


def canonical_fill_model(raw: str) -> str:
    value = raw.strip().lower().replace("_", "-")
    if value in {"lobster-level", "lobster"}:
        return "lobster_level"
    if value in {"queue", "queue-aware"}:
        return "queue"
    if value == "trade-through":
        return "trade-through"
    raise ValueError(f"Unsupported fill model: {raw!r}")


def date_from_path(path: Path) -> str | None:
    iso_match = ISO_DATE_RE.search(path.name)
    if iso_match:
        return iso_match.group(1)
    match = DATE_RE.search(path.name)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def date_from_mbo(path: Path, mbo: pd.DataFrame) -> str:
    parsed = date_from_path(path)
    if parsed is not None:
        return parsed
    if "ts_event" in mbo.columns and not mbo.empty:
        return str(pd.Timestamp(mbo["ts_event"].iloc[0]).date())
    return path.stem


def finite_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def list_input_files(data_dir: Path) -> list[Path]:
    patterns = ("*.mbo.dbn.zst", "*.dbn.zst", "*.parquet", "*.csv", "*.csv.gz")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(data_dir.rglob(pattern))
    seen: set[Path] = set()
    unique: list[Path] = []
    def sort_key(path: Path) -> tuple[str, str]:
        return (date_from_path(path) or "9999-99-99", str(path))

    for path in sorted(files, key=sort_key):
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def filter_files_by_date(files: list[Path], start: str | None, end: str | None) -> list[Path]:
    selected: list[Path] = []
    for path in files:
        date = date_from_path(path)
        if date is None:
            continue
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        selected.append(path)
    return selected


def select_split_files(
    *,
    data_dir: Path,
    files: list[Path],
    sample_data: Path | None,
    start: str | None,
    end: str | None,
    start_index: int,
    end_index: int,
    split_name: str,
) -> list[Path]:
    if sample_data is not None:
        return [sample_data]
    split_dir = data_dir / split_name
    base_files = list_input_files(split_dir) if split_dir.is_dir() else files
    if start is not None or end is not None:
        selected = filter_files_by_date(base_files, start, end)
    elif split_dir.is_dir():
        selected = base_files
    else:
        if start_index < 1 or end_index < start_index:
            raise ValueError(f"Invalid {split_name} index range: {start_index}-{end_index}")
        selected = base_files[start_index - 1 : end_index]
    if not selected:
        raise FileNotFoundError(f"No {split_name} input files selected.")
    return selected


BaselineInput = Union[Path, LobsterDayFiles]


def input_date(item: BaselineInput) -> str | None:
    if isinstance(item, LobsterDayFiles):
        return item.date.isoformat()
    return date_from_path(item)


def input_date_range(items: Sequence[BaselineInput]) -> tuple[str | None, str | None]:
    dates = [value for value in (input_date(item) for item in items) if value is not None]
    if not dates:
        return (None, None)
    return (min(dates), max(dates))


def filter_lobster_by_date(
    pairs: list[LobsterDayFiles],
    start: str | None,
    end: str | None,
) -> list[LobsterDayFiles]:
    selected: list[LobsterDayFiles] = []
    for pair in pairs:
        day = pair.date.isoformat()
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        selected.append(pair)
    return selected


def lobster_split_dir(data_dir: Path, split_name: str) -> Path | None:
    for dirname in LOBSTER_SPLIT_DIR_ALIASES[split_name]:
        candidate = data_dir / dirname
        if candidate.is_dir():
            return candidate
    return None


def select_lobster_split(
    *,
    data_dir: Path | None = None,
    pairs: list[LobsterDayFiles] | None = None,
    sample_data_dir: Path | None,
    levels: int = DEFAULT_LOBSTER_LEVELS,
    start: str | None,
    end: str | None,
    split_name: str,
) -> list[LobsterDayFiles]:
    if sample_data_dir is not None:
        pairs = pairs if pairs is not None else discover_lobster_pairs(sample_data_dir, levels=levels)
        if not pairs:
            raise FileNotFoundError(f"No LOBSTER fixture files found in {sample_data_dir}.")
        return pairs

    if data_dir is not None:
        split_dir = lobster_split_dir(data_dir, split_name)
        if split_dir is not None:
            selected = discover_lobster_pairs(split_dir, levels=levels)
            if start is not None or end is not None:
                selected = filter_lobster_by_date(selected, start, end)
            if not selected:
                raise FileNotFoundError(f"No {split_name} LOBSTER files selected in {split_dir}.")
            return selected

    if pairs is None:
        if data_dir is None:
            raise ValueError("select_lobster_split requires pairs or data_dir.")
        pairs = discover_lobster_pairs(data_dir, levels=levels)

    default_start, default_end = LOBSTER_SPLIT_DATES[split_name]
    selected = filter_lobster_by_date(pairs, start or default_start, end or default_end)
    if not selected:
        raise FileNotFoundError(
            f"No {split_name} LOBSTER files selected for {start or default_start} to {end or default_end}."
        )
    return selected


def detect_data_format(args: argparse.Namespace) -> str:
    if args.data_format != "auto":
        return args.data_format
    if args.sample_data is not None:
        return "databento"
    search_dir = args.sample_data_dir or args.data_dir
    if is_lobster_dir(search_dir, levels=args.lobster_levels):
        return "lobster"
    return "databento"


def prepare_day(item: BaselineInput, args: argparse.Namespace) -> DayData:
    if isinstance(item, LobsterDayFiles):
        day = load_lobster_day(
            item,
            dt_seconds=args.dt_seconds,
            cache_dir=args.cache_dir,
        )
        return DayData(
            date=item.date.isoformat(),
            mbo=None,
            mbo_rth=None,
            bbo=day.bbo,
            extrema=day.trade_extrema,
            trade_flow=day.trade_flow,
            data_format="lobster",
        )

    path = item
    mbo = load_mbo(path)
    bbo_all = build_sampled_bbo(mbo, dt=pd.Timedelta(seconds=args.dt_seconds))
    bbo = filter_regular_hours(bbo_all)
    mbo_rth = filter_regular_hours(mbo)
    if bbo.empty:
        raise RuntimeError(f"No regular-hours BBO samples for {path}")
    extrema = interval_trade_extrema(mbo_rth, bbo.index, args.dt_seconds)
    trade_flow = build_interval_trade_flow(mbo_rth, bbo.index, args.dt_seconds)
    return DayData(
        date=date_from_mbo(path, mbo),
        mbo=mbo,
        mbo_rth=mbo_rth,
        bbo=bbo,
        extrema=extrema,
        trade_flow=trade_flow,
        data_format="databento",
    )


def calibrate_day(day: DayData, args: argparse.Namespace) -> dict[str, Any]:
    deltas = np.arange(args.delta_start_ticks, args.delta_end_ticks + 1) * args.tick_size
    sigma = estimate_sigma_from_mid(day.bbo)
    if args.fill_model == "lobster_level":
        calibration = calibrate_lobster_arrival_rates(
            bbo=day.bbo,
            trade_flow=day.trade_flow,
            deltas=deltas,
            dt_seconds=args.dt_seconds,
            tick_size=args.tick_size,
            order_size=args.order_size,
        )
    elif args.fill_model == "queue":
        if day.mbo is None:
            raise ValueError("The queue fill model requires Databento MBO data. Use --fill-model lobster_level for LOBSTER.")
        calibration = calibrate_queue_arrival_rates(
            mbo=day.mbo,
            bbo=day.bbo,
            trade_flow=day.trade_flow,
            deltas=deltas,
            dt_seconds=args.dt_seconds,
            tick_size=args.tick_size,
            order_size=args.order_size,
        )
    else:
        calibration = calibrate_arrival_rates(day.bbo, day.extrema, deltas=deltas, dt_seconds=args.dt_seconds)
    return {
        "date": day.date,
        "data_format": day.data_format,
        "sigma": float(sigma),
        "A": float(calibration.A),
        "k": float(calibration.k),
        "A_ask": float(calibration.A_ask),
        "k_ask": float(calibration.k_ask),
        "A_bid": float(calibration.A_bid),
        "k_bid": float(calibration.k_bid),
        "num_bbo_samples": int(len(day.bbo)),
        "num_arrival_bins": int(np.isfinite(calibration.lambda_ask).sum() + np.isfinite(calibration.lambda_bid).sum()),
    }


def validate_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Calibrated {name} must be finite and positive, got {value!r}")


def _calibrate_day_worker(payload: tuple[BaselineInput, argparse.Namespace]) -> dict[str, Any]:
    item, args = payload
    return calibrate_day(prepare_day(item, args), args)


def calibrate_training(files: Sequence[BaselineInput], args: argparse.Namespace) -> CalibrationResult:
    rows: list[dict[str, Any]] = []
    if args.workers > 1 and len(files) > 1:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                rows.extend(pool.map(_calibrate_day_worker, [(item, args) for item in files]))
        except PermissionError as exc:
            print(f"Process workers unavailable ({exc}); falling back to serial calibration.")
            for path in files:
                rows.append(_calibrate_day_worker((path, args)))
    else:
        for path in files:
            rows.append(_calibrate_day_worker((path, args)))

    sigma = finite_median(row["sigma"] for row in rows)
    arrival_a = finite_median(row["A"] for row in rows)
    arrival_k = finite_median(row["k"] for row in rows)
    validate_positive("sigma", sigma)
    validate_positive("A", arrival_a)
    validate_positive("k", arrival_k)

    gamma_seed = choose_gamma_for_target_skew(
        sigma=sigma,
        horizon_seconds=args.horizon_seconds,
        q_target=max(1, args.inventory_limit),
        target_total_skew=args.target_skew,
    )
    return CalibrationResult(
        sigma=sigma,
        A=arrival_a,
        k=arrival_k,
        data_format=args.resolved_data_format,
        lobster_levels=args.lobster_levels if args.resolved_data_format == "lobster" else None,
        num_train_days=len(rows),
        sigma_method="winsorized_midprice_diff",
        arrival_fit_method=f"log_linear_{args.fill_model}",
        decision_interval_seconds=args.dt_seconds,
        fill_model=args.fill_model,
        order_size=args.order_size,
        inventory_limit=args.inventory_limit,
        tick_size=args.tick_size,
        horizon_mode=args.as_horizon_mode,
        horizon_seconds=args.horizon_seconds,
        gamma_seed=float(gamma_seed),
        train_start=getattr(args, "train_start_actual", None),
        train_end=getattr(args, "train_end_actual", None),
        validation_start=getattr(args, "validation_start_actual", None),
        validation_end=getattr(args, "validation_end_actual", None),
        test_start=getattr(args, "test_start_actual", None),
        test_end=getattr(args, "test_end_actual", None),
        daily_calibrations=[
            {key: value for key, value in row.items() if isinstance(value, (float, int, str))}
            for row in rows
        ],
    )


def parse_gamma_values(raw: str | None, calibration: CalibrationResult, args: argparse.Namespace) -> list[tuple[str, float]]:
    if raw:
        out: list[tuple[str, float]] = []
        for idx, item in enumerate(raw.replace(";", ",").split(",")):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                label, raw_value = item.split("=", 1)
                label = label.strip()
            else:
                label = f"gamma_{idx}"
                raw_value = item
            gamma = float(raw_value.strip())
            if not np.isfinite(gamma) or gamma < 0:
                raise ValueError(f"Invalid gamma value: {item!r}")
            out.append((label, gamma))
        if not out:
            raise ValueError("--gamma-values did not contain any gamma values.")
        return out

    gamma_1tick = args.tick_size / (
        max(1e-12, args.q_ref * calibration.sigma * calibration.sigma * args.horizon_seconds)
    )
    return [
        ("zero", 0.0),
        ("calibrated", calibration.gamma_seed),
        ("one_tick", float(gamma_1tick)),
        ("four_tick", float(4.0 * gamma_1tick)),
    ]


def make_params(gamma: float, calibration: CalibrationResult, args: argparse.Namespace) -> ASParams:
    return ASParams(
        gamma=gamma,
        sigma=calibration.sigma,
        k=calibration.k,
        A=calibration.A,
        horizon_seconds=args.horizon_seconds,
        dt_seconds=args.dt_seconds,
        tick_size=args.tick_size,
        order_size=args.order_size,
        q_max=args.inventory_limit,
        rolling_horizon=args.as_horizon_mode == "rolling",
    )


def calibration_from_json(path: Path) -> CalibrationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {field.name for field in fields(CalibrationResult)}
    missing = sorted(names - set(payload))
    if missing:
        raise ValueError(f"Calibration file missing fields: {missing}")
    return CalibrationResult(**{name: payload[name] for name in names})


def _close_float(left: float, right: float, *, tol: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def validate_resume_calibration(calibration: CalibrationResult, args: argparse.Namespace) -> None:
    checks: list[tuple[str, Any, Any, bool]] = [
        ("data_format", calibration.data_format, args.resolved_data_format, calibration.data_format == args.resolved_data_format),
        ("tick_size", calibration.tick_size, args.tick_size, _close_float(calibration.tick_size, args.tick_size)),
        (
            "decision_interval_seconds",
            calibration.decision_interval_seconds,
            args.dt_seconds,
            _close_float(calibration.decision_interval_seconds, args.dt_seconds),
        ),
        ("order_size", calibration.order_size, args.order_size, calibration.order_size == args.order_size),
        (
            "inventory_limit",
            calibration.inventory_limit,
            args.inventory_limit,
            calibration.inventory_limit == args.inventory_limit,
        ),
        ("horizon_mode", calibration.horizon_mode, args.as_horizon_mode, calibration.horizon_mode == args.as_horizon_mode),
        (
            "horizon_seconds",
            calibration.horizon_seconds,
            args.horizon_seconds,
            _close_float(calibration.horizon_seconds, args.horizon_seconds),
        ),
        ("fill_model", calibration.fill_model, args.fill_model, calibration.fill_model == args.fill_model),
        (
            "lobster_levels",
            calibration.lobster_levels,
            args.lobster_levels if args.resolved_data_format == "lobster" else None,
            calibration.lobster_levels == (args.lobster_levels if args.resolved_data_format == "lobster" else None),
        ),
    ]
    mismatches = [
        f"{name}: calibration={stored!r}, current={current!r}"
        for name, stored, current, ok in checks
        if not ok
    ]
    if mismatches:
        raise ValueError("--resume-from-calibration does not match current run config: " + "; ".join(mismatches))


def load_episode_manifest(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    manifest = pd.read_csv(path)
    if "date" not in manifest.columns:
        raise ValueError("--episode-starts-csv must contain a date column.")
    if "start_ts" not in manifest.columns and "start_index" not in manifest.columns:
        raise ValueError("--episode-starts-csv must contain start_ts or start_index.")
    return manifest


def episode_slices(day: DayData, args: argparse.Namespace, manifest: pd.DataFrame | None) -> list[tuple[str, int, int]]:
    n = len(day.bbo)
    steps = args.episode_steps
    if manifest is not None:
        rows = manifest[manifest["date"].astype(str).eq(day.date)]
        out: list[tuple[str, int, int]] = []
        for idx, row in enumerate(rows.itertuples(index=False)):
            if hasattr(row, "start_index") and pd.notna(row.start_index):
                start = int(row.start_index)
            else:
                start_ts = pd.Timestamp(row.start_ts)
                if start_ts.tzinfo is None:
                    start_ts = start_ts.tz_localize("UTC")
                else:
                    start_ts = start_ts.tz_convert("UTC")
                start = int(day.bbo.index.searchsorted(start_ts))
            end = min(start + steps, n)
            if end > start:
                episode_id = str(getattr(row, "episode_id", f"{day.date}_{idx:04d}"))
                out.append((episode_id, start, end))
        return out

    full_count = n // steps
    if full_count == 0:
        return [(f"{day.date}_0000", 0, n)]
    return [(f"{day.date}_{idx:04d}", idx * steps, (idx + 1) * steps) for idx in range(full_count)]


def validate_replay_frame(df: pd.DataFrame, *, inventory_limit: int) -> None:
    required = ["mid", "bid", "ask", "spread", "inventory", "cash", "pnl"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Replay output missing columns: {missing}")
    for column in required:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"Replay output has non-finite values in {column}")
    if (df["ask"] <= df["bid"]).any():
        raise ValueError("Replay output contains crossed quotes.")
    if (df["spread"] <= 0).any():
        raise ValueError("Replay output contains non-positive spreads.")
    if (df["inventory"].abs() > inventory_limit).any():
        raise ValueError("Replay output exceeds the configured inventory limit.")


def summarize_replay_episode(
    df: pd.DataFrame,
    *,
    strategy: str,
    split: str,
    date: str,
    episode_id: str,
    gamma: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validate_replay_frame(df, inventory_limit=args.inventory_limit)
    pv = portfolio_value(df["cash"], df["inventory"], df["mid"])
    bid_orders = float(df["quote_bid"].sum())
    ask_orders = float(df["quote_ask"].sum())
    bid_fills = float(df["bid_fill"].sum())
    ask_fills = float(df["ask_fill"].sum())
    fills_by_step = df["bid_fill"].to_numpy(dtype=float) + df["ask_fill"].to_numpy(dtype=float)
    quote_skew = df["reservation"].to_numpy(dtype=float) - df["mid"].to_numpy(dtype=float)

    return {
        "strategy": strategy,
        "baseline_family": "classical_control",
        "split": split,
        "date": date,
        "episode_id": episode_id,
        "gamma": float(gamma),
        "data_format": args.resolved_data_format,
        "fill_model": args.fill_model,
        "lobster_levels": args.lobster_levels if args.resolved_data_format == "lobster" else 0,
        "horizon_mode": args.as_horizon_mode,
        "horizon_seconds": args.horizon_seconds,
        "decision_interval_ms": args.decision_interval_ms,
        "order_size": args.order_size,
        "inventory_limit": args.inventory_limit,
        "tick_size": args.tick_size,
        "final_pv": float(pv[-1]),
        "mean_pv": float(np.mean(pv)),
        "std_pv": float(np.std(pv, ddof=1)) if len(pv) > 1 else 0.0,
        "max_drawdown": max_drawdown(pv),
        "final_inventory": float(df["inventory"].iloc[-1]),
        "avg_inventory": float(df["inventory"].mean()),
        "avg_q2": avg_q2(df["inventory"]),
        "max_abs_inventory": max_abs_inventory(df["inventory"]),
        "num_bid_orders": int(bid_orders),
        "num_ask_orders": int(ask_orders),
        "num_bid_fills": int(bid_fills),
        "num_ask_fills": int(ask_fills),
        "fill_rate": fill_rate(bid_fills + ask_fills, bid_orders + ask_orders),
        "order_submission_rate": order_submission_rate(bid_orders + ask_orders, len(df)),
        "no_trade_rate": no_trade_rate(fills_by_step),
        "mean_spread": float(df["spread"].mean()),
        "mean_abs_skew": float(np.mean(np.abs(quote_skew))),
    }


def summarize_no_trade_episode(
    bbo_slice: pd.DataFrame,
    *,
    split: str,
    date: str,
    episode_id: str,
    gamma: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "strategy": "no_trade",
        "baseline_family": "passive_control",
        "split": split,
        "date": date,
        "episode_id": episode_id,
        "gamma": float(gamma),
        "data_format": args.resolved_data_format,
        "fill_model": "none",
        "lobster_levels": args.lobster_levels if args.resolved_data_format == "lobster" else 0,
        "horizon_mode": args.as_horizon_mode,
        "horizon_seconds": args.horizon_seconds,
        "decision_interval_ms": args.decision_interval_ms,
        "order_size": args.order_size,
        "inventory_limit": args.inventory_limit,
        "tick_size": args.tick_size,
        "final_pv": 0.0,
        "mean_pv": 0.0,
        "std_pv": 0.0,
        "max_drawdown": 0.0,
        "final_inventory": 0.0,
        "avg_inventory": 0.0,
        "avg_q2": 0.0,
        "max_abs_inventory": 0.0,
        "num_bid_orders": 0,
        "num_ask_orders": 0,
        "num_bid_fills": 0,
        "num_ask_fills": 0,
        "fill_rate": 0.0,
        "order_submission_rate": 0.0,
        "no_trade_rate": 1.0,
        "mean_spread": 0.0,
        "mean_abs_skew": 0.0,
    }


def replay_strategy_for_label(strategy: str) -> str:
    if strategy == "as_classical_control":
        return "inventory"
    if strategy == "symmetric_mm":
        return "symmetric"
    raise ValueError(f"Unsupported replay strategy label: {strategy!r}")


def slice_lobster_trade_flow(trade_flow: LobsterTradeFlow, start: int, end: int) -> LobsterTradeFlow:
    return LobsterTradeFlow(
        buy_aggressor=trade_flow.buy_aggressor[start:end],
        sell_aggressor=trade_flow.sell_aggressor[start:end],
    )


def simulate_episode_replay(
    day: DayData,
    *,
    start: int,
    end: int,
    params: ASParams,
    args: argparse.Namespace,
    replay_strategy: str,
) -> pd.DataFrame:
    """Replay one episode with local cash, inventory, and AS elapsed time."""
    bbo_slice = day.bbo.iloc[start:end]
    if bbo_slice.empty:
        return pd.DataFrame()

    if args.fill_model == "lobster_level":
        flow_slice = slice_lobster_trade_flow(day.trade_flow, start, end)
        return simulate_lobster_replay(
            bbo_slice,
            flow_slice,
            params,
            strategy=replay_strategy,
        )

    extrema_slice = day.extrema.iloc[start:end]
    return simulate_replay(
        bbo_slice,
        extrema_slice,
        params,
        strategy=replay_strategy,
        fill_model=args.fill_model,
        mbo=day.mbo,
        trade_flow=None,
    )


def _evaluate_file(
    item: BaselineInput,
    *,
    split: str,
    gamma: float,
    calibration: CalibrationResult,
    args: argparse.Namespace,
    strategies: tuple[str, ...],
    manifest: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = make_params(gamma, calibration, args)
    day = prepare_day(item, args)
    slices = episode_slices(day, args, manifest)

    for episode_id, start, end in slices:
        for strategy in strategies:
            if strategy == "no_trade":
                continue
            replay_strategy = replay_strategy_for_label(strategy)
            episode = simulate_episode_replay(
                day,
                start=start,
                end=end,
                params=params,
                args=args,
                replay_strategy=replay_strategy,
            )
            if not episode.empty:
                rows.append(
                    summarize_replay_episode(
                        episode,
                        strategy=strategy,
                        split=split,
                        date=day.date,
                        episode_id=episode_id,
                        gamma=gamma,
                        args=args,
                    )
                )
        if "no_trade" in strategies:
            rows.append(
                summarize_no_trade_episode(
                    day.bbo.iloc[start:end],
                    split=split,
                    date=day.date,
                    episode_id=episode_id,
                    gamma=gamma,
                    args=args,
                )
            )

    return rows


def _evaluate_file_worker(
    payload: tuple[BaselineInput, str, float, CalibrationResult, argparse.Namespace, tuple[str, ...], pd.DataFrame | None],
) -> list[dict[str, Any]]:
    item, split, gamma, calibration, args, strategies, manifest = payload
    return _evaluate_file(
        item,
        split=split,
        gamma=gamma,
        calibration=calibration,
        args=args,
        strategies=strategies,
        manifest=manifest,
    )


def evaluate_files(
    files: Sequence[BaselineInput],
    *,
    split: str,
    gamma: float,
    calibration: CalibrationResult,
    args: argparse.Namespace,
    strategies: tuple[str, ...],
    manifest: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    payloads = [(item, split, gamma, calibration, args, strategies, manifest) for item in files]
    if args.workers > 1 and len(files) > 1:
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for part in pool.map(_evaluate_file_worker, payloads):
                    rows.extend(part)
        except PermissionError as exc:
            print(f"Process workers unavailable ({exc}); falling back to serial {split} evaluation.")
            for payload in payloads:
                rows.extend(_evaluate_file_worker(payload))
    else:
        for payload in payloads:
            rows.extend(_evaluate_file_worker(payload))
    return pd.DataFrame(rows)


def build_gamma_selection(
    validation_files: Sequence[BaselineInput],
    gamma_grid: list[tuple[str, float]],
    calibration: CalibrationResult,
    args: argparse.Namespace,
    manifest: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    threshold = args.q_ref * args.q_ref
    for label, gamma in gamma_grid:
        metrics = evaluate_files(
            validation_files,
            split="validation",
            gamma=gamma,
            calibration=calibration,
            args=args,
            strategies=("as_classical_control",),
            manifest=manifest,
        )
        if metrics.empty:
            raise RuntimeError(f"No validation metrics produced for gamma={label}")
        mean_final_pv = float(metrics["final_pv"].mean())
        mean_avg_q2 = float(metrics["avg_q2"].mean())
        feasible = mean_avg_q2 <= threshold
        rows.append(
            {
                "gamma_label": label,
                "gamma": float(gamma),
                "mean_final_pv": mean_final_pv,
                "std_final_pv": float(metrics["final_pv"].std(ddof=1)) if len(metrics) > 1 else 0.0,
                "mean_avg_q2": mean_avg_q2,
                "mean_max_abs_inventory": float(metrics["max_abs_inventory"].mean()),
                "mean_fill_rate": float(metrics["fill_rate"].mean()),
                "mean_spread": float(metrics["mean_spread"].mean()),
                "mean_skew_abs": float(metrics["mean_abs_skew"].mean()),
                "inventory_risk_threshold": float(threshold),
                "selection_score": mean_final_pv if feasible else float("-inf"),
                "selected": False,
            }
        )

    selection = pd.DataFrame(rows)
    feasible = selection[np.isfinite(selection["selection_score"])]
    if feasible.empty:
        order = selection.sort_values(["mean_avg_q2", "mean_final_pv"], ascending=[True, False])
    else:
        order = feasible.sort_values(["selection_score", "mean_avg_q2"], ascending=[False, True])
    selected_index = int(order.index[0])
    selection.loc[selected_index, "selected"] = True
    return selection


def aggregate_daily_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    groups = metrics.groupby(["date", "strategy"], sort=False)
    rows: list[dict[str, Any]] = []
    for (date, strategy), part in groups:
        rows.append(
            {
                "date": date,
                "strategy": strategy,
                "episodes": int(len(part)),
                "mean_final_pv": float(part["final_pv"].mean()),
                "std_final_pv": float(part["final_pv"].std(ddof=1)) if len(part) > 1 else 0.0,
                "mean_avg_q2": float(part["avg_q2"].mean()),
                "mean_max_abs_inventory": float(part["max_abs_inventory"].mean()),
                "mean_fill_rate": float(part["fill_rate"].mean()),
                "mean_order_submission_rate": float(part["order_submission_rate"].mean()),
                "mean_no_trade_rate": float(part["no_trade_rate"].mean()),
                "mean_spread": float(part["mean_spread"].mean()),
                "mean_abs_skew": float(part["mean_abs_skew"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_summary(metrics: pd.DataFrame, calibration: CalibrationResult, selected_gamma: float) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metadata": {
            "selected_gamma": selected_gamma,
            "calibration": {k: v for k, v in asdict(calibration).items() if k != "daily_calibrations"},
        },
        "strategies": {},
    }
    for strategy, part in metrics.groupby("strategy", sort=False):
        summary["strategies"][strategy] = {
            "episodes": int(len(part)),
            "mean_final_pv": float(part["final_pv"].mean()),
            "std_final_pv": float(part["final_pv"].std(ddof=1)) if len(part) > 1 else 0.0,
            "mean_avg_q2": float(part["avg_q2"].mean()),
            "max_abs_inventory": float(part["max_abs_inventory"].max()),
            "mean_fill_rate": float(part["fill_rate"].mean()),
            "mean_order_submission_rate": float(part["order_submission_rate"].mean()),
            "mean_no_trade_rate": float(part["no_trade_rate"].mean()),
        }
    return summary


def build_run_manifest(
    args: argparse.Namespace,
    *,
    calibration: CalibrationResult,
    selected_gamma: float,
    gamma_selection: pd.DataFrame,
    test_strategies: tuple[str, ...],
) -> dict[str, Any]:
    selected = gamma_selection[gamma_selection["selected"]].iloc[0].to_dict()
    return {
        "baseline_family": "classical_control",
        "selected_gamma": float(selected_gamma),
        "selected_gamma_label": str(selected["gamma_label"]),
        "calibration_source": "resumed" if args.resume_from_calibration is not None else "fresh",
        "resume_from_calibration": None if args.resume_from_calibration is None else str(args.resume_from_calibration),
        "splits": {
            "train": {"start": args.train_start_actual, "end": args.train_end_actual},
            "validation": {"start": args.validation_start_actual, "end": args.validation_end_actual},
            "test": {"start": args.test_start_actual, "end": args.test_end_actual},
        },
        "config": {
            "data_format": args.resolved_data_format,
            "fill_model": args.fill_model,
            "lobster_levels": args.lobster_levels if args.resolved_data_format == "lobster" else None,
            "decision_interval_ms": args.decision_interval_ms,
            "episode_steps": args.episode_steps,
            "order_size": args.order_size,
            "inventory_limit": args.inventory_limit,
            "tick_size": args.tick_size,
            "horizon_mode": args.as_horizon_mode,
            "horizon_seconds": args.horizon_seconds,
            "q_ref": args.q_ref,
            "target_skew": args.target_skew,
            "test_strategies": list(test_strategies),
            "include_symmetric": bool(args.include_symmetric),
        },
        "calibration": {k: v for k, v in asdict(calibration).items() if k != "daily_calibrations"},
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_rgb_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in pixels)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _blank_canvas(width: int = 900, height: int = 560) -> list[list[tuple[int, int, int]]]:
    return [[(255, 255, 255) for _ in range(width)] for _ in range(height)]


def _draw_rect(
    pixels: list[list[tuple[int, int, int]]],
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
) -> None:
    height = len(pixels)
    width = len(pixels[0])
    for y in range(max(0, top), min(height, bottom + 1)):
        row = pixels[y]
        for x in range(max(0, left), min(width, right + 1)):
            row[x] = color


def _draw_line(
    pixels: list[list[tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    height = len(pixels)
    width = len(pixels[0])
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            pixels[y0][x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _scale(value: float, low: float, high: float, out_low: int, out_high: int) -> int:
    if not np.isfinite(value) or not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return (out_low + out_high) // 2
    ratio = (value - low) / (high - low)
    ratio = min(1.0, max(0.0, ratio))
    return int(round(out_low + ratio * (out_high - out_low)))


def render_fallback_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    grouped = metrics.groupby("strategy", sort=False).agg(
        mean_final_pv=("final_pv", "mean"),
        mean_avg_q2=("avg_q2", "mean"),
        fill_rate=("fill_rate", "mean"),
        order_submission_rate=("order_submission_rate", "mean"),
        no_trade_rate=("no_trade_rate", "mean"),
    )
    colors = [(31, 119, 180), (44, 160, 44), (214, 39, 40), (148, 103, 189)]

    pixels = _blank_canvas()
    left, right, top, bottom = 90, 830, 50, 490
    _draw_line(pixels, left, bottom, right, bottom, (30, 30, 30))
    _draw_line(pixels, left, top, left, bottom, (30, 30, 30))
    x_values = grouped["mean_avg_q2"].to_numpy(dtype=float)
    y_values = grouped["mean_final_pv"].to_numpy(dtype=float)
    x_low, x_high = 0.0, float(np.nanmax(x_values) * 1.05) if len(x_values) else 1.0
    y_low = float(min(np.nanmin(y_values), 0.0)) if len(y_values) else 0.0
    y_high = float(max(np.nanmax(y_values), 0.0)) if len(y_values) else 1.0
    if y_high <= y_low:
        y_high = y_low + 1.0
    for idx, (_, row) in enumerate(grouped.iterrows()):
        x = _scale(float(row["mean_avg_q2"]), x_low, x_high, left + 8, right - 8)
        y = _scale(float(row["mean_final_pv"]), y_low, y_high, bottom - 8, top + 8)
        color = colors[idx % len(colors)]
        _draw_rect(pixels, x - 7, y - 7, x + 7, y + 7, color)
        _draw_rect(pixels, right - 150, top + idx * 24, right - 132, top + idx * 24 + 14, color)
    _write_rgb_png(output_dir / "risk_return_frontier.png", pixels)

    pixels = _blank_canvas()
    left, right, top, bottom = 80, 840, 50, 500
    _draw_line(pixels, left, bottom, right, bottom, (30, 30, 30))
    _draw_line(pixels, left, top, left, bottom, (30, 30, 30))
    rate_cols = ["no_trade_rate", "order_submission_rate", "fill_rate"]
    rate_colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44)]
    strategies = list(grouped.index)
    group_width = (right - left) / max(1, len(strategies))
    bar_width = max(8, int(group_width / 5))
    max_rate = float(max(1.0, grouped[rate_cols].to_numpy(dtype=float).max()))
    for i, strategy in enumerate(strategies):
        center = int(left + group_width * (i + 0.5))
        for j, column in enumerate(rate_cols):
            value = float(grouped.loc[strategy, column])
            bar_height = int((bottom - top - 12) * min(max(value / max_rate, 0.0), 1.0))
            x0 = center + (j - 1) * (bar_width + 4)
            _draw_rect(pixels, x0, bottom - bar_height, x0 + bar_width, bottom, rate_colors[j])
    for j, color in enumerate(rate_colors):
        _draw_rect(pixels, right - 150, top + j * 24, right - 132, top + j * 24 + 14, color)
    _write_rgb_png(output_dir / "activity_vs_inventory_risk.png", pixels)


def render_default_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - depends on optional matplotlib
        render_fallback_plots(metrics, output_dir)
        return

    grouped = metrics.groupby("strategy", sort=False).agg(
        mean_final_pv=("final_pv", "mean"),
        mean_avg_q2=("avg_q2", "mean"),
        fill_rate=("fill_rate", "mean"),
        order_submission_rate=("order_submission_rate", "mean"),
        no_trade_rate=("no_trade_rate", "mean"),
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(grouped["mean_avg_q2"], grouped["mean_final_pv"], s=70)
    for strategy, row in grouped.iterrows():
        ax.annotate(strategy, (row["mean_avg_q2"], row["mean_final_pv"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Average squared inventory")
    ax.set_ylabel("Mean final portfolio value")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "risk_return_frontier.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(grouped))
    width = 0.26
    ax.bar(x - width, grouped["no_trade_rate"], width, label="no-trade")
    ax.bar(x, grouped["order_submission_rate"], width, label="submission")
    ax.bar(x + width, grouped["fill_rate"], width, label="fill")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=20, ha="right")
    ax.set_ylim(0, max(1.0, float(grouped[["no_trade_rate", "order_submission_rate", "fill_rate"]].max().max()) * 1.1))
    ax.set_ylabel("Rate")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "activity_vs_inventory_risk.png", dpi=160)
    plt.close(fig)


def render_debug_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        for filename in DEBUG_FIGURES:
            (output_dir / filename).write_bytes(PLACEHOLDER_PNG)
        (output_dir / "as_debug_report.html").write_text(
            "<!doctype html><html><body><h1>AS Debug Report</h1><p>matplotlib unavailable; placeholder images written.</p></body></html>",
            encoding="utf-8",
        )
        return

    by_gamma = metrics.groupby("gamma", sort=True).agg(
        final_pv=("final_pv", "mean"),
        avg_q2=("avg_q2", "mean"),
        fill_rate=("fill_rate", "mean"),
        mean_abs_skew=("mean_abs_skew", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(by_gamma.index, by_gamma["final_pv"], marker="o")
    ax.set_xlabel("gamma")
    ax.set_ylabel("Mean final PV")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "gamma_sweep_full.png", dpi=160)
    plt.close(fig)
    for filename in DEBUG_FIGURES:
        path = output_dir / filename
        if not path.exists():
            path.write_bytes(PLACEHOLDER_PNG)

    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>AS Debug Report</title></head>
<body><h1>AS Debug Report</h1><p>Debug artifacts were generated explicitly with --debug-plots.</p>
<img src="gamma_sweep_full.png" alt="Gamma sweep"></body></html>
"""
    (output_dir / "as_debug_report.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cleaned Avellaneda-Stoikov baseline pipeline.")
    parser.add_argument("--data-format", choices=["auto", "lobster", "databento"], default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lobster_amzn_10"))
    parser.add_argument("--sample-data", type=Path, default=None, help="Use one tiny fixture for train/validation/test.")
    parser.add_argument("--sample-data-dir", type=Path, default=None, help="Use a tiny LOBSTER fixture directory for train/validation/test.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/AS"))
    parser.add_argument("--baselines-output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--debug-dir", type=Path, default=Path("debug/as"))
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional compact LOBSTER cache directory, e.g. data/cache/as_lobster_10.")
    parser.add_argument("--resume-from-calibration", type=Path, default=None, help="Skip train calibration and reuse an existing as_calibration.json.")
    parser.add_argument("--workers", type=int, default=1, help="Reserved day-level worker count for batch runs.")
    parser.add_argument("--lobster-levels", type=int, default=DEFAULT_LOBSTER_LEVELS)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-start", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--train-start-index", type=int, default=1)
    parser.add_argument("--train-end-index", type=int, default=74)
    parser.add_argument("--val-start-index", type=int, default=75)
    parser.add_argument("--val-end-index", type=int, default=99)
    parser.add_argument("--test-start-index", type=int, default=100)
    parser.add_argument("--test-end-index", type=int, default=123)
    parser.add_argument(
        "--fill-model",
        choices=["lobster_level", "lobster-level", "queue", "queue_aware", "queue-aware", "trade-through", "trade_through"],
        default="lobster_level",
    )
    parser.add_argument("--order-size", type=int, default=10)
    parser.add_argument("--inventory-limit", type=int, default=200)
    parser.add_argument("--decision-interval-ms", type=int, default=250)
    parser.add_argument("--episode-steps", type=int, default=64)
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--as-horizon-mode", choices=["rolling", "finite_episode"], default="finite_episode")
    parser.add_argument("--as-horizon-seconds", type=float, default=None)
    parser.add_argument("--target-skew", type=float, default=0.05)
    parser.add_argument("--q-ref", type=float, default=20.0)
    parser.add_argument("--delta-start-ticks", type=int, default=1)
    parser.add_argument("--delta-end-ticks", type=int, default=20)
    parser.add_argument("--gamma-values", default=None)
    parser.add_argument("--episode-starts-csv", type=Path, default=None)
    parser.add_argument("--include-symmetric", action="store_true", help="Include the old symmetric market-maker benchmark row.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--debug-plots", action="store_true")
    args = parser.parse_args()

    args.fill_model = canonical_fill_model(args.fill_model)
    if args.sample_data is not None and args.sample_data_dir is not None:
        raise SystemExit("Use only one of --sample-data or --sample-data-dir.")
    if args.resume_from_calibration is not None and not args.resume_from_calibration.exists():
        raise SystemExit(f"Calibration file not found: {args.resume_from_calibration}")
    args.dt_seconds = args.decision_interval_ms / 1000.0
    args.horizon_seconds = (
        float(args.as_horizon_seconds)
        if args.as_horizon_seconds is not None
        else float(args.episode_steps * args.dt_seconds)
    )
    if args.order_size <= 0:
        raise SystemExit("--order-size must be positive.")
    if args.inventory_limit < 0:
        raise SystemExit("--inventory-limit must be non-negative.")
    if args.episode_steps <= 0:
        raise SystemExit("--episode-steps must be positive.")
    if args.delta_start_ticks < 1 or args.delta_end_ticks < args.delta_start_ticks:
        raise SystemExit("Invalid delta tick range.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.lobster_levels <= 0:
        raise SystemExit("--lobster-levels must be positive.")
    return args


def main() -> None:
    args = parse_args()
    args.resolved_data_format = detect_data_format(args)
    if args.resolved_data_format == "lobster" and args.fill_model == "queue":
        args.fill_model = "lobster_level"
    if args.resolved_data_format == "databento" and args.fill_model == "lobster_level":
        args.fill_model = "queue"

    if args.sample_data is not None and not args.sample_data.exists():
        raise SystemExit(f"Sample data not found: {args.sample_data}")
    if args.sample_data_dir is not None and not args.sample_data_dir.is_dir():
        raise SystemExit(f"Sample data directory not found: {args.sample_data_dir}")

    train_files: Sequence[BaselineInput] = []
    if args.resolved_data_format == "lobster":
        lobster_dir = args.sample_data_dir or args.data_dir
        if args.resume_from_calibration is None:
            train_files = select_lobster_split(
                data_dir=lobster_dir,
                sample_data_dir=args.sample_data_dir,
                levels=args.lobster_levels,
                start=args.train_start,
                end=args.train_end,
                split_name="train",
            )
        validation_files: Sequence[BaselineInput] = select_lobster_split(
            data_dir=lobster_dir,
            sample_data_dir=args.sample_data_dir,
            levels=args.lobster_levels,
            start=args.val_start,
            end=args.val_end,
            split_name="validation",
        )
        test_files: Sequence[BaselineInput] = select_lobster_split(
            data_dir=lobster_dir,
            sample_data_dir=args.sample_data_dir,
            levels=args.lobster_levels,
            start=args.test_start,
            end=args.test_end,
            split_name="test",
        )
    else:
        files = [] if args.sample_data is not None else list_input_files(args.data_dir)
        if args.sample_data is None and not files:
            raise SystemExit(f"No input files found under {args.data_dir}")
        if args.resume_from_calibration is None:
            train_files = select_split_files(
                data_dir=args.data_dir,
                files=files,
                sample_data=args.sample_data,
                start=args.train_start,
                end=args.train_end,
                start_index=args.train_start_index,
                end_index=args.train_end_index,
                split_name="train",
            )
        validation_files = select_split_files(
            data_dir=args.data_dir,
            files=files,
            sample_data=args.sample_data,
            start=args.val_start,
            end=args.val_end,
            start_index=args.val_start_index,
            end_index=args.val_end_index,
            split_name="validation",
        )
        test_files = select_split_files(
            data_dir=args.data_dir,
            files=files,
            sample_data=args.sample_data,
            start=args.test_start,
            end=args.test_end,
            start_index=args.test_start_index,
            end_index=args.test_end_index,
            split_name="test",
        )

    args.train_start_actual, args.train_end_actual = input_date_range(train_files)
    args.validation_start_actual, args.validation_end_actual = input_date_range(validation_files)
    args.test_start_actual, args.test_end_actual = input_date_range(test_files)
    manifest = load_episode_manifest(args.episode_starts_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.baselines_output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from_calibration is not None:
        print(f"Loading AS calibration from {args.resume_from_calibration} ...")
        calibration = calibration_from_json(args.resume_from_calibration)
        validate_resume_calibration(calibration, args)
        args.train_start_actual = calibration.train_start
        args.train_end_actual = calibration.train_end
    else:
        print(
            "Calibrating AS baseline on training data "
            f"({args.resolved_data_format}, {len(train_files)} days, "
            f"{args.train_start_actual} to {args.train_end_actual}) ..."
        )
        calibration = calibrate_training(train_files, args)
    write_json(args.output_dir / "as_calibration.json", asdict(calibration))

    print(
        "Selecting gamma on validation data "
        f"({len(validation_files)} days, {args.validation_start_actual} to {args.validation_end_actual}) ..."
    )
    gamma_grid = parse_gamma_values(args.gamma_values, calibration, args)
    gamma_selection = build_gamma_selection(validation_files, gamma_grid, calibration, args, manifest)
    gamma_selection.to_csv(args.output_dir / "as_gamma_selection.csv", index=False)
    selected_gamma = float(gamma_selection.loc[gamma_selection["selected"], "gamma"].iloc[0])

    test_strategies: tuple[str, ...] = ("as_classical_control", "no_trade")
    if args.include_symmetric:
        test_strategies = ("as_classical_control", "symmetric_mm", "no_trade")

    print(
        "Evaluating selected AS classical-control and no-trade baselines on test data "
        f"({len(test_files)} days, {args.test_start_actual} to {args.test_end_actual}) ..."
    )
    test_metrics = evaluate_files(
        test_files,
        split="test",
        gamma=selected_gamma,
        calibration=calibration,
        args=args,
        strategies=test_strategies,
        manifest=manifest,
    )
    daily = aggregate_daily_metrics(test_metrics)
    summary = build_summary(test_metrics, calibration, selected_gamma)
    manifest_payload = build_run_manifest(
        args,
        calibration=calibration,
        selected_gamma=selected_gamma,
        gamma_selection=gamma_selection,
        test_strategies=test_strategies,
    )

    test_metrics.to_csv(args.output_dir / "as_test_metrics.csv", index=False)
    daily.to_csv(args.output_dir / "as_test_daily_metrics.csv", index=False)
    write_json(args.output_dir / "as_test_summary.json", summary)
    test_metrics.to_csv(args.output_dir / "as_for_rl_comparison.csv", index=False)
    write_json(args.output_dir / "as_run_manifest.json", manifest_payload)

    no_trade = test_metrics[test_metrics["strategy"].eq("no_trade")].reset_index(drop=True)
    no_trade.to_csv(args.baselines_output_dir / "no_trade_for_rl_comparison.csv", index=False)
    write_json(
        args.baselines_output_dir / "no_trade_test_summary.json",
        {
            "metadata": summary["metadata"],
            "strategies": {"no_trade": summary["strategies"].get("no_trade", {})},
        },
    )

    if not args.no_plots:
        render_default_plots(test_metrics, args.output_dir)
    if args.debug_plots:
        render_debug_plots(test_metrics, args.debug_dir)

    print(f"AS baseline complete. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
