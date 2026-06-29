from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .calibration import calibrate_ac_params, load_calibration, save_calibration
from .data import load_lobster_orderbook, read_snapshots, read_table, reconstruct_mbo_snapshots, standardize_snapshot_columns, write_table
from .metrics import add_policy_names, aggregate_by_policy, paired_policy_difference
from .plots import plot_frontier, plot_impact_curve, plot_schedules, plot_slippage_distribution
from .replay import evaluate_ac_grid
from .schema import EpisodeSpec


def _parse_float_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_grid(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def cmd_normalize_lobster(args: argparse.Namespace) -> None:
    df = load_lobster_orderbook(
        args.orderbook,
        message_path=args.messages,
        levels=args.levels,
        price_scale=args.price_scale,
        header=args.header,
    )
    write_table(df, args.out)
    print(f"Wrote {len(df):,} canonical snapshots to {args.out}")


def cmd_standardize_snapshots(args: argparse.Namespace) -> None:
    raw = read_table(args.input)
    df = standardize_snapshot_columns(
        raw,
        price_scale=args.price_scale,
        depth_levels=args.levels,
        timestamp_col=args.timestamp_col,
    )
    write_table(df, args.out)
    print(f"Wrote {len(df):,} canonical snapshots to {args.out}")

def cmd_reconstruct_mbo(args: argparse.Namespace) -> None:
    raw = read_table(args.input)
    df = reconstruct_mbo_snapshots(
        raw,
        depth_levels=args.levels,
        price_scale=args.price_scale,
        sample_every_events=args.sample_every_events,
        max_events=args.max_events,
    )
    write_table(df, args.out)
    print(f"Reconstructed {len(df):,} canonical snapshots to {args.out}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    snapshots = read_snapshots(args.snapshots, price_scale=args.price_scale, depth_levels=args.levels)
    q_grid = _parse_int_grid(args.q_grid)
    params, impact_curve = calibrate_ac_params(
        snapshots,
        q_grid=q_grid,
        step_stride_rows=args.messages_per_step,
        depth=args.levels,
        max_rows=args.max_rows,
        random_seed=args.seed,
    )
    save_calibration(params, impact_curve, args.out_dir)
    plot_impact_curve(impact_curve, out_path=Path(args.out_dir) / "book_walk_impact_curve.png")
    print(json.dumps(params.to_dict(), indent=2))
    print(f"Wrote calibration outputs to {args.out_dir}")


def _episode_spec_from_args(args: argparse.Namespace) -> EpisodeSpec:
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


def cmd_evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = read_snapshots(args.snapshots, price_scale=args.price_scale, depth_levels=args.levels)
    spec = _episode_spec_from_args(args)
    kappa_T_grid = _parse_float_grid(args.kappa_grid)

    metrics_df, fills_df, plan_df = evaluate_ac_grid(
        snapshots,
        spec=spec,
        kappa_T_grid=kappa_T_grid,
        depth=args.levels,
        max_episodes=args.max_episodes,
        return_fills=args.save_fills,
        carry_unfilled=not args.no_carry_unfilled,
    )
    summary_df = aggregate_by_policy(metrics_df, n_boot=args.n_boot, seed=args.seed)
    summary_df = add_policy_names(summary_df)

    metrics_df.to_csv(out_dir / "episode_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "policy_summary.csv", index=False)
    plan_df.to_csv(out_dir / "episode_plan.csv", index=False)
    if fills_df is not None:
        fills_df.to_csv(out_dir / "fills.csv", index=False)

    # Paired differences against TWAP when present.
    if any(abs(float(k)) < 1e-12 for k in kappa_T_grid):
        diffs = []
        for k in kappa_T_grid:
            if abs(float(k)) < 1e-12:
                continue
            diffs.append(
                paired_policy_difference(
                    metrics_df,
                    metric="slippage_ticks_per_share",
                    baseline_kappa_T=0.0,
                    compare_kappa_T=float(k),
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
            )
        if diffs:
            pd.DataFrame(diffs).to_csv(out_dir / "paired_slippage_vs_twap.csv", index=False)

    plot_schedules(
        task_size=spec.task_size,
        n_steps=spec.episode_length,
        kappa_T_grid=kappa_T_grid,
        out_path=out_dir / "ac_schedules.png",
    )
    plot_frontier(summary_df, out_path=out_dir / "risk_cost_frontier.png")
    plot_slippage_distribution(metrics_df, out_path=out_dir / "slippage_distribution.png")

    print(summary_df.to_string(index=False))
    print(f"Wrote evaluation outputs to {out_dir}")


def cmd_plot_schedules(args: argparse.Namespace) -> None:
    plot_schedules(
        task_size=args.task_size,
        n_steps=args.episode_length,
        kappa_T_grid=_parse_float_grid(args.kappa_grid),
        out_path=args.out,
    )
    print(f"Wrote schedule plot to {args.out}")


def cmd_make_synthetic(args: argparse.Namespace) -> None:
    """Create a tiny synthetic canonical snapshot file for smoke tests."""
    rng = np.random.default_rng(args.seed)
    n = args.rows
    levels = args.levels
    mid = 200.0 + np.cumsum(rng.normal(0, 0.01, size=n))
    spread = np.full(n, 0.02)
    data = {"timestamp": np.arange(n, dtype=np.int64), "best_bid": mid - spread / 2, "best_ask": mid + spread / 2}
    for lvl in range(1, levels + 1):
        offset = spread / 2 + (lvl - 1) * 0.01
        data[f"ask_price_{lvl}"] = mid + offset
        data[f"bid_price_{lvl}"] = mid - offset
        data[f"ask_size_{lvl}"] = rng.integers(100, 1000, size=n)
        data[f"bid_size_{lvl}"] = rng.integers(100, 1000, size=n)
    df = pd.DataFrame(data)
    df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2
    df["spread"] = df["best_ask"] - df["best_bid"]
    write_table(df, args.out)
    print(f"Wrote synthetic snapshots to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Almgren-Chriss execution benchmark CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("normalize-lobster", help="Convert LOBSTER orderbook CSV to canonical snapshots")
    p.add_argument("--orderbook", required=True, help="LOBSTER orderbook CSV")
    p.add_argument("--messages", default=None, help="Optional LOBSTER message CSV for timestamps")
    p.add_argument("--levels", type=int, default=None, help="Depth levels to keep")
    p.add_argument("--price-scale", type=float, default=10000.0, help="LOBSTER price scaling divisor")
    p.add_argument("--header", action="store_true", help="Input orderbook already has a header")
    p.add_argument("--out", required=True, help="Output .parquet or .csv")
    p.set_defaults(func=cmd_normalize_lobster)

    p = sub.add_parser("standardize-snapshots", help="Canonicalize generic/Databento-like snapshot CSV or Parquet")
    p.add_argument("--input", required=True, help="Input snapshot CSV/Parquet")
    p.add_argument("--out", required=True, help="Output canonical .parquet or .csv")
    p.add_argument("--levels", type=int, default=None, help="Depth levels to keep")
    p.add_argument("--price-scale", type=float, default=1.0, help="Price scaling divisor if needed")
    p.add_argument("--timestamp-col", default=None, help="Explicit timestamp column name")
    p.set_defaults(func=cmd_standardize_snapshots)

    p = sub.add_parser("reconstruct-mbo", help="Reconstruct top-K snapshots from Databento-like raw MBO events")
    p.add_argument("--input", required=True, help="Raw MBO event CSV/Parquet")
    p.add_argument("--out", required=True, help="Output canonical .parquet or .csv")
    p.add_argument("--levels", type=int, default=10, help="Depth levels to output")
    p.add_argument("--price-scale", type=float, default=1.0, help="Price scaling divisor if needed")
    p.add_argument("--sample-every-events", type=int, default=100, help="Emit one snapshot every N events")
    p.add_argument("--max-events", type=int, default=None, help="Optional event limit for pilot runs")
    p.set_defaults(func=cmd_reconstruct_mbo)

    p = sub.add_parser("calibrate", help="Estimate volatility, half-spread, and eta from train snapshots")
    p.add_argument("--snapshots", required=True, help="Canonical train snapshots CSV/Parquet")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--q-grid", default="10,20,50,100,200,400,600")
    p.add_argument("--messages-per-step", type=int, default=100)
    p.add_argument("--levels", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=250000)
    p.add_argument("--price-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("evaluate", help="Run AC/TWAP replay evaluation on validation or test snapshots")
    p.add_argument("--snapshots", required=True, help="Canonical snapshots CSV/Parquet")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--kappa-grid", default="0,0.5,1,2,4")
    p.add_argument("--task-size", type=int, default=600)
    p.add_argument("--episode-length", type=int, default=64)
    p.add_argument("--messages-per-step", type=int, default=100)
    p.add_argument("--episode-start-frequency-steps", type=int, default=64)
    p.add_argument("--lot-size", type=int, default=10)
    p.add_argument("--tick-size", type=float, default=0.01)
    p.add_argument("--directions", choices=["random", "alternating", "buy", "sell"], default="random")
    p.add_argument("--levels", type=int, default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--price-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--save-fills", action="store_true")
    p.add_argument("--no-carry-unfilled", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("plot-schedules", help="Plot AC remaining-quantity trajectories")
    p.add_argument("--out", required=True)
    p.add_argument("--kappa-grid", default="0,0.5,1,2,4")
    p.add_argument("--task-size", type=int, default=600)
    p.add_argument("--episode-length", type=int, default=64)
    p.set_defaults(func=cmd_plot_schedules)

    p = sub.add_parser("make-synthetic", help="Create tiny synthetic canonical snapshots for smoke testing")
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=10000)
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_make_synthetic)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

