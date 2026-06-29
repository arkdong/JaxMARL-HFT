from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .calibration import calibrate_ac_params
from .data import load_lobster_orderbook, read_snapshots, write_table
from .fast_calibration import calibrate_ac_params_fast
from .fast_replay import evaluate_ac_grid_fast
from .metrics import add_policy_names, aggregate_by_policy
from .plots import plot_frontier, plot_impact_curve, plot_schedules, plot_slippage_distribution
from .replay import evaluate_ac_grid
from .schema import ACParams, EpisodeSpec


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
LOBSTER_FILE_RE = re.compile(
    r"^(?P<symbol>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<start>\d+)_(?P<end>\d+)_(?P<kind>message|orderbook)_"
    r"(?P<levels>\d+)\.csv$"
)


@dataclass(frozen=True)
class LobsterDayFiles:
    date: date
    message_path: Path
    orderbook_path: Path
    symbol: str
    levels: int


def progress(message: str) -> None:
    print(message, flush=True)


def elapsed_seconds(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.1f}s"


def parse_grid(raw: str, cast: type = float) -> list[Any]:
    values = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return [cast(value) for value in values]


def _parse_lobster_filename(path: Path) -> dict[str, Any]:
    match = LOBSTER_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"not a LOBSTER filename: {path}")
    groups = match.groupdict()
    return {
        "symbol": groups["symbol"],
        "date": date.fromisoformat(groups["date"]),
        "kind": groups["kind"],
        "levels": int(groups["levels"]),
    }


def discover_lobster_pairs(data_dir: Path | str, *, levels: int) -> list[LobsterDayFiles]:
    root = Path(data_dir)
    messages: dict[date, Path] = {}
    orderbooks: dict[date, Path] = {}
    metadata: dict[date, dict[str, Any]] = {}

    for path in sorted(root.glob(f"*_message_{levels}.csv")):
        info = _parse_lobster_filename(path)
        day = info["date"]
        messages[day] = path
        metadata[day] = info
    for path in sorted(root.glob(f"*_orderbook_{levels}.csv")):
        info = _parse_lobster_filename(path)
        day = info["date"]
        orderbooks[day] = path
        metadata[day] = info

    days = sorted(set(messages) | set(orderbooks))
    missing = [
        day.isoformat()
        for day in days
        if day not in messages or day not in orderbooks
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete LOBSTER message/orderbook pairs in {root}: {missing}")

    pairs = []
    for day in days:
        info = metadata[day]
        pairs.append(
            LobsterDayFiles(
                date=day,
                message_path=messages[day],
                orderbook_path=orderbooks[day],
                symbol=str(info["symbol"]),
                levels=int(info["levels"]),
            )
        )
    return pairs


def lobster_split_dir(data_dir: Path, split_name: str) -> Path | None:
    for dirname in LOBSTER_SPLIT_DIR_ALIASES[split_name]:
        candidate = data_dir / dirname
        if candidate.is_dir():
            return candidate
    return None


def filter_lobster_by_date(
    pairs: Sequence[LobsterDayFiles],
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


def select_lobster_split(
    *,
    data_dir: Path,
    levels: int,
    split_name: str,
    start: str | None = None,
    end: str | None = None,
) -> list[LobsterDayFiles]:
    split_dir = lobster_split_dir(data_dir, split_name)
    if split_dir is not None:
        pairs = discover_lobster_pairs(split_dir, levels=levels)
        selected = filter_lobster_by_date(pairs, start, end) if (start or end) else pairs
        if not selected:
            raise FileNotFoundError(f"No {split_name} LOBSTER files selected in {split_dir}.")
        return selected

    pairs = discover_lobster_pairs(data_dir, levels=levels)
    default_start, default_end = LOBSTER_SPLIT_DATES[split_name]
    selected = filter_lobster_by_date(pairs, start or default_start, end or default_end)
    if not selected:
        raise FileNotFoundError(
            f"No {split_name} LOBSTER files selected for {start or default_start} to {end or default_end}."
        )
    return selected


def input_date_range(days: Sequence[LobsterDayFiles]) -> tuple[str | None, str | None]:
    if not days:
        return None, None
    values = [day.date.isoformat() for day in days]
    return min(values), max(values)


def cache_path_for_day(day: LobsterDayFiles, args: argparse.Namespace) -> Path | None:
    if args.cache_dir is None:
        return None
    return (
        Path(args.cache_dir)
        / "ac_lobster_snapshots"
        / f"{day.symbol}_{day.date.isoformat()}_levels{args.lobster_levels}_ps{int(args.price_scale)}.parquet"
    )


def load_day_snapshots(day: LobsterDayFiles, args: argparse.Namespace) -> pd.DataFrame:
    cache_path = cache_path_for_day(day, args)
    if cache_path is not None and cache_path.exists():
        snapshots = read_snapshots(cache_path, depth_levels=args.lobster_levels)
    else:
        snapshots = load_lobster_orderbook(
            day.orderbook_path,
            message_path=day.message_path,
            levels=args.lobster_levels,
            price_scale=args.price_scale,
        )
        if cache_path is not None:
            write_table(snapshots, cache_path)
    snapshots.insert(0, "date", day.date.isoformat())
    return snapshots


def make_episode_spec(args: argparse.Namespace) -> EpisodeSpec:
    return EpisodeSpec(
        task_size=args.task_size,
        episode_length=args.episode_length,
        messages_per_step=args.messages_per_step,
        episode_start_frequency_steps=args.episode_start_frequency_steps,
        lot_size=args.lot_size,
        tick_size=args.tick_size,
        directions=args.directions,
        random_seed=args.seed,
    )


def should_use_fast_replay(engine: str, *, save_fills: bool) -> bool:
    if engine == "slow":
        return False
    if engine == "fast" and save_fills:
        raise SystemExit("--engine fast does not support --save-fills; use --engine slow for fill-level output.")
    return not save_fills


def calibrate_lobster_train(days: Sequence[LobsterDayFiles], args: argparse.Namespace) -> tuple[ACParams, pd.DataFrame, pd.DataFrame]:
    q_grid = parse_grid(args.q_grid, int)
    daily_rows: list[dict[str, Any]] = []
    impact_frames: list[pd.DataFrame] = []
    calibration_engine = getattr(args, "calibration_engine", "fast")
    calibrator = calibrate_ac_params_fast if calibration_engine == "fast" else calibrate_ac_params

    for idx, day in enumerate(days, start=1):
        started_at = time.perf_counter()
        progress(f"[ac calibration:{calibration_engine}] {idx}/{len(days)} start {day.date.isoformat()}")
        snapshots = load_day_snapshots(day, args)
        params, impact_curve = calibrator(
            snapshots,
            q_grid=q_grid,
            step_stride_rows=args.messages_per_step,
            depth=args.lobster_levels,
            max_rows=args.max_rows_calibration,
            random_seed=args.seed,
        )
        daily_rows.append(
            {
                "date": day.date.isoformat(),
                "n_obs": int(params.n_obs),
                "sigma_step": float(params.sigma_step),
                "half_spread": float(params.half_spread),
                "eta_ac": float(params.eta_ac),
            }
        )
        impact_curve.insert(0, "date", day.date.isoformat())
        impact_frames.append(impact_curve)
        progress(
            f"[ac calibration:{calibration_engine}] {idx}/{len(days)} complete {day.date.isoformat()} "
            f"sigma_step={params.sigma_step:.6g} half_spread={params.half_spread:.6g} "
            f"eta_ac={params.eta_ac:.6g} ({elapsed_seconds(started_at)})"
        )

    daily = pd.DataFrame(daily_rows)
    params = ACParams(
        sigma_step=float(daily["sigma_step"].median()),
        half_spread=float(daily["half_spread"].median()),
        eta_ac=float(daily["eta_ac"].median()),
        gamma_ac=0.0,
        q_grid=list(map(int, q_grid)),
        n_obs=int(daily["n_obs"].sum()),
        notes="Median daily AC calibration from LOBSTER train split; gamma_ac kept at zero.",
    )
    impact = pd.concat(impact_frames, ignore_index=True) if impact_frames else pd.DataFrame()
    return params, daily, impact


def strategy_name(kappa_t: float) -> str:
    if abs(float(kappa_t)) < 1e-12:
        return "twap"
    return f"ac_kappa_{float(kappa_t):g}"


def evaluate_lobster_days(
    days: Sequence[LobsterDayFiles],
    *,
    split: str,
    kappa_grid: Sequence[float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    spec = make_episode_spec(args)
    metric_frames: list[pd.DataFrame] = []
    plan_frames: list[pd.DataFrame] = []
    fill_frames: list[pd.DataFrame] = []
    engine = getattr(args, "engine", "auto")
    use_fast = should_use_fast_replay(engine, save_fills=args.save_fills)
    effective_engine = "fast" if use_fast else "slow"

    for idx, day in enumerate(days, start=1):
        started_at = time.perf_counter()
        progress(
            f"[ac {split}:{effective_engine}] {idx}/{len(days)} start {day.date.isoformat()} "
            f"kappa_grid={','.join(f'{k:g}' for k in kappa_grid)}"
        )
        snapshots = load_day_snapshots(day, args)
        if use_fast:
            metrics_df, fills_df, plan_df = evaluate_ac_grid_fast(
                snapshots,
                spec=spec,
                kappa_T_grid=kappa_grid,
                depth=args.lobster_levels,
                max_episodes=args.max_episodes,
                carry_unfilled=not args.no_carry_unfilled,
            )
        else:
            metrics_df, fills_df, plan_df = evaluate_ac_grid(
                snapshots,
                spec=spec,
                kappa_T_grid=kappa_grid,
                depth=args.lobster_levels,
                max_episodes=args.max_episodes,
                return_fills=args.save_fills,
                carry_unfilled=not args.no_carry_unfilled,
            )
        metrics_df.insert(0, "strategy", metrics_df["kappa_T"].map(strategy_name))
        metrics_df.insert(1, "baseline_family", "classical_control")
        metrics_df.insert(2, "split", split)
        metrics_df.insert(3, "date", day.date.isoformat())
        metrics_df["local_episode_id"] = metrics_df["episode_id"].astype(int)
        metrics_df["episode_key"] = metrics_df["date"] + "_" + metrics_df["local_episode_id"].map(lambda x: f"{x:04d}")
        metrics_df["episode_id"] = metrics_df["episode_key"]
        metrics_df["data_format"] = "lobster"
        metrics_df["lobster_levels"] = int(args.lobster_levels)
        metrics_df["tick_size"] = float(args.tick_size)
        metrics_df["lot_size"] = int(args.lot_size)
        metric_frames.append(metrics_df)

        plan_df.insert(0, "split", split)
        plan_df.insert(1, "date", day.date.isoformat())
        plan_df["local_episode_id"] = plan_df["episode_id"].astype(int)
        plan_df["episode_key"] = plan_df["date"] + "_" + plan_df["local_episode_id"].map(lambda x: f"{x:04d}")
        plan_df["episode_id"] = plan_df["episode_key"]
        plan_frames.append(plan_df)

        if fills_df is not None and not fills_df.empty:
            fills_df.insert(0, "split", split)
            fills_df.insert(1, "date", day.date.isoformat())
            fills_df["local_episode_id"] = fills_df["episode_id"].astype(int)
            fills_df["episode_key"] = fills_df["date"] + "_" + fills_df["local_episode_id"].map(lambda x: f"{x:04d}")
            fills_df["episode_id"] = fills_df["episode_key"]
            fill_frames.append(fills_df)

        progress(
            f"[ac {split}:{effective_engine}] {idx}/{len(days)} complete {day.date.isoformat()} "
            f"episodes={len(plan_df)} metric_rows={len(metrics_df)} ({elapsed_seconds(started_at)})"
        )

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    plans = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()
    fills = pd.concat(fill_frames, ignore_index=True) if fill_frames else None
    return metrics, plans, fills


def build_kappa_selection(validation_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = validation_metrics.groupby("kappa_T", sort=True).agg(
        strategy=("strategy", "first"),
        mean_slippage_ticks_per_share=("slippage_ticks_per_share", "mean"),
        std_slippage_ticks_per_share=("slippage_ticks_per_share", "std"),
        mean_completion_rate=("completion_rate", "mean"),
        mean_unfinished=("unfinished", "mean"),
        mean_avg_sq_remaining=("avg_sq_remaining", "mean"),
        episodes=("episode_key", "nunique"),
    )
    selection = grouped.reset_index()
    selection["std_slippage_ticks_per_share"] = selection["std_slippage_ticks_per_share"].fillna(0.0)
    order = selection.sort_values(
        ["mean_unfinished", "mean_completion_rate", "mean_slippage_ticks_per_share", "mean_avg_sq_remaining"],
        ascending=[True, False, True, True],
    )
    selected_idx = int(order.index[0])
    selection["selected"] = False
    selection.loc[selected_idx, "selected"] = True
    return selection


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_manifest(
    args: argparse.Namespace,
    *,
    train_days: Sequence[LobsterDayFiles],
    validation_days: Sequence[LobsterDayFiles],
    test_days: Sequence[LobsterDayFiles],
    selected_kappa: float,
    params: ACParams,
) -> dict[str, Any]:
    train_start, train_end = input_date_range(train_days)
    val_start, val_end = input_date_range(validation_days)
    test_start, test_end = input_date_range(test_days)
    return {
        "baseline": "almgren_chriss",
        "baseline_family": "classical_control",
        "data_format": "lobster",
        "data_dir": str(args.data_dir),
        "cache_dir": None if args.cache_dir is None else str(args.cache_dir),
        "lobster_levels": int(args.lobster_levels),
        "splits": {
            "train": {"days": len(train_days), "start": train_start, "end": train_end},
            "validation": {"days": len(validation_days), "start": val_start, "end": val_end},
            "test": {"days": len(test_days), "start": test_start, "end": test_end},
        },
        "selected_kappa_T": float(selected_kappa),
        "kappa_grid": [float(k) for k in parse_grid(args.kappa_grid, float)],
        "test_kappa_grid": [0.0, float(selected_kappa)] if abs(float(selected_kappa)) > 1e-12 else [0.0],
        "episode": asdict(make_episode_spec(args)),
        "price_scale": float(args.price_scale),
        "max_rows_calibration": args.max_rows_calibration,
        "max_episodes": args.max_episodes,
        "evaluation_engine": getattr(args, "engine", "auto"),
        "calibration_engine": getattr(args, "calibration_engine", "fast"),
        "save_fills": bool(args.save_fills),
        "calibration": params.to_dict(),
    }


def run_lobster_pipeline(args: argparse.Namespace) -> None:
    if args.data_format not in {"auto", "lobster"}:
        raise SystemExit("AC LOBSTER pipeline currently supports --data-format lobster or auto.")
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_days = select_lobster_split(
        data_dir=args.data_dir,
        levels=args.lobster_levels,
        split_name="train",
        start=args.train_start,
        end=args.train_end,
    )
    validation_days = select_lobster_split(
        data_dir=args.data_dir,
        levels=args.lobster_levels,
        split_name="validation",
        start=args.val_start,
        end=args.val_end,
    )
    test_days = select_lobster_split(
        data_dir=args.data_dir,
        levels=args.lobster_levels,
        split_name="test",
        start=args.test_start,
        end=args.test_end,
    )

    train_start, train_end = input_date_range(train_days)
    val_start, val_end = input_date_range(validation_days)
    test_start, test_end = input_date_range(test_days)

    progress(f"Calibrating AC on LOBSTER train data ({len(train_days)} days, {train_start} to {train_end}) ...")
    params, daily_calibration, impact_curve = calibrate_lobster_train(train_days, args)
    write_json(args.output_dir / "ac_calibration.json", params.to_dict())
    daily_calibration.to_csv(args.output_dir / "ac_daily_calibration.csv", index=False)
    impact_curve.to_csv(args.output_dir / "ac_book_walk_impact_curve.csv", index=False)

    kappa_grid = parse_grid(args.kappa_grid, float)
    progress(f"Selecting AC kappa_T on LOBSTER validation data ({len(validation_days)} days, {val_start} to {val_end}) ...")
    validation_metrics, validation_plan, validation_fills = evaluate_lobster_days(
        validation_days,
        split="validation",
        kappa_grid=kappa_grid,
        args=args,
    )
    validation_summary = add_policy_names(aggregate_by_policy(validation_metrics, n_boot=args.n_boot, seed=args.seed))
    kappa_selection = build_kappa_selection(validation_metrics)
    selected_kappa = float(kappa_selection.loc[kappa_selection["selected"], "kappa_T"].iloc[0])

    validation_metrics.to_csv(args.output_dir / "ac_validation_metrics.csv", index=False)
    validation_summary.to_csv(args.output_dir / "ac_validation_policy_summary.csv", index=False)
    validation_plan.to_csv(args.output_dir / "ac_validation_episode_plan.csv", index=False)
    kappa_selection.to_csv(args.output_dir / "ac_kappa_selection.csv", index=False)
    if validation_fills is not None:
        validation_fills.to_csv(args.output_dir / "ac_validation_fills.csv", index=False)

    test_kappa_grid = [0.0] if abs(selected_kappa) < 1e-12 else [0.0, selected_kappa]
    progress(
        "Evaluating selected AC/TWAP policies on LOBSTER test data "
        f"({len(test_days)} days, {test_start} to {test_end}, selected_kappa_T={selected_kappa:g}) ..."
    )
    test_metrics, test_plan, test_fills = evaluate_lobster_days(
        test_days,
        split="test",
        kappa_grid=test_kappa_grid,
        args=args,
    )
    test_summary = add_policy_names(aggregate_by_policy(test_metrics, n_boot=args.n_boot, seed=args.seed))
    manifest = build_manifest(
        args,
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        selected_kappa=selected_kappa,
        params=params,
    )

    test_metrics.to_csv(args.output_dir / "ac_test_metrics.csv", index=False)
    test_summary.to_csv(args.output_dir / "ac_test_policy_summary.csv", index=False)
    test_plan.to_csv(args.output_dir / "ac_test_episode_plan.csv", index=False)
    test_metrics.to_csv(args.output_dir / "ac_for_rl_comparison.csv", index=False)
    write_json(args.output_dir / "ac_run_manifest.json", manifest)
    if test_fills is not None:
        test_fills.to_csv(args.output_dir / "ac_test_fills.csv", index=False)

    if not args.no_plots:
        plot_impact_curve(impact_curve, out_path=args.output_dir / "book_walk_impact_curve.png")
        plot_schedules(
            task_size=args.task_size,
            n_steps=args.episode_length,
            kappa_T_grid=test_kappa_grid,
            out_path=args.output_dir / "ac_schedules.png",
        )
        plot_frontier(test_summary, out_path=args.output_dir / "risk_cost_frontier.png")
        plot_slippage_distribution(test_metrics, out_path=args.output_dir / "slippage_distribution.png")

    progress(f"AC LOBSTER pipeline complete. Outputs written to {args.output_dir}")
