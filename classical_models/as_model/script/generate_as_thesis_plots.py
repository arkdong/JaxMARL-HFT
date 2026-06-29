#!/usr/bin/env python3
"""
Generate thesis-ready plots for the Avellaneda-Stoikov classical benchmark.

Expected inputs:
    results/AS/as_gamma_selection.csv
    results/AS/as_test_metrics.csv
    results/AS/as_test_daily_metrics.csv
    results/AS/as_test_summary.json
    results/baselines/no_trade_for_rl_comparison.csv   optional

Outputs:
    results/AS/thesis_plots/
        01_as_gamma_selection_frontier.png/pdf
        02_as_gamma_selection_by_gamma.png/pdf
        03_as_test_risk_return_frontier.png/pdf
        04_as_test_activity_rates.png/pdf
        05_as_daily_mean_final_pv.png/pdf
        06_as_daily_avg_q2.png/pdf
        07_as_episode_final_pv_distribution.png/pdf
        08_as_daily_fill_vs_pv.png/pdf
        09_as_episode_q2_vs_pv.png/pdf
        10_as_episode_drawdown_distribution.png/pdf
        as_test_strategy_summary.csv
        as_daily_strategy_summary.csv
        plot_manifest.json

Notes:
    - No seaborn dependency.
    - Handles optional symmetric_mm rows if the AS benchmark is rerun with --include-symmetric.
    - Uses daily aggregation where possible for uncertainty because episodes inside one day are not independent.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# -----------------------------
# General helpers
# -----------------------------

STRATEGY_LABELS = {
    "as_classical_control": "AS classical",
    "symmetric_mm": "Symmetric MM",
    "no_trade": "No-trade",
}


def label_strategy(strategy: str) -> str:
    return STRATEGY_LABELS.get(str(strategy), str(strategy).replace("_", " "))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File exists but is empty: {path}")


def read_csv_required(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV has no rows: {path}")
    return df


def read_json_required(path: Path) -> dict:
    require_file(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_numeric(df: pd.DataFrame, skip: Iterable[str] = ()) -> pd.DataFrame:
    out = df.copy()
    skip = set(skip)
    for col in out.columns:
        if col in skip:
            continue
        try:
            out[col] = pd.to_numeric(out[col])
        except (TypeError, ValueError):
            pass
    return out


def selected_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        written.append(str(path))
    plt.close(fig)
    return written


def annotate_points(ax: plt.Axes, x: pd.Series, y: pd.Series, labels: pd.Series) -> None:
    for xi, yi, lab in zip(x, y, labels):
        if pd.notna(xi) and pd.notna(yi):
            ax.annotate(str(lab), (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)


def clean_strategy_order(strategies: Iterable[str]) -> list[str]:
    preferred = ["no_trade", "symmetric_mm", "as_classical_control"]
    found = list(dict.fromkeys(strategies))
    ordered = [s for s in preferred if s in found]
    ordered.extend([s for s in found if s not in ordered])
    return ordered


def daily_ci_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over test days, not episodes, to avoid pretending that adjacent intraday episodes
    are independent observations.
    """
    rows = []
    for strategy, part in daily.groupby("strategy", sort=False):
        part = part.copy()
        n_days = int(part["date"].nunique())
        pv_std = float(part["mean_final_pv"].std(ddof=1)) if n_days > 1 else 0.0
        q2_std = float(part["mean_avg_q2"].std(ddof=1)) if n_days > 1 else 0.0
        rows.append(
            {
                "strategy": strategy,
                "strategy_label": label_strategy(strategy),
                "days": n_days,
                "daily_mean_final_pv": float(part["mean_final_pv"].mean()),
                "daily_se_final_pv": pv_std / math.sqrt(n_days) if n_days > 0 else np.nan,
                "daily_ci95_final_pv": 1.96 * pv_std / math.sqrt(n_days) if n_days > 0 else np.nan,
                "daily_mean_avg_q2": float(part["mean_avg_q2"].mean()),
                "daily_se_avg_q2": q2_std / math.sqrt(n_days) if n_days > 0 else np.nan,
                "daily_ci95_avg_q2": 1.96 * q2_std / math.sqrt(n_days) if n_days > 0 else np.nan,
                "daily_mean_fill_rate": float(part["mean_fill_rate"].mean()),
                "daily_mean_order_submission_rate": float(part["mean_order_submission_rate"].mean()),
                "daily_mean_no_trade_rate": float(part["mean_no_trade_rate"].mean()),
                "daily_mean_spread": float(part["mean_spread"].mean()),
                "daily_mean_abs_skew": float(part["mean_abs_skew"].mean()),
            }
        )
    return pd.DataFrame(rows)


def episode_strategy_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, part in metrics.groupby("strategy", sort=False):
        final_pv = part["final_pv"].astype(float)
        avg_q2 = part["avg_q2"].astype(float)
        max_drawdown = part["max_drawdown"].astype(float)
        rows.append(
            {
                "strategy": strategy,
                "strategy_label": label_strategy(strategy),
                "episodes": int(len(part)),
                "mean_final_pv": float(final_pv.mean()),
                "std_final_pv": float(final_pv.std(ddof=1)) if len(part) > 1 else 0.0,
                "median_final_pv": float(final_pv.median()),
                "p05_final_pv": float(final_pv.quantile(0.05)),
                "p95_final_pv": float(final_pv.quantile(0.95)),
                "prob_final_pv_negative": float((final_pv < 0).mean()),
                "mean_avg_q2": float(avg_q2.mean()),
                "median_avg_q2": float(avg_q2.median()),
                "p95_avg_q2": float(avg_q2.quantile(0.95)),
                "max_abs_inventory": float(part["max_abs_inventory"].max()),
                "mean_fill_rate": float(part["fill_rate"].mean()),
                "mean_order_submission_rate": float(part["order_submission_rate"].mean()),
                "mean_no_trade_rate": float(part["no_trade_rate"].mean()),
                "mean_max_drawdown": float(max_drawdown.mean()),
                "p05_max_drawdown": float(max_drawdown.quantile(0.05)),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Plot functions
# -----------------------------

def plot_gamma_selection_frontier(
    gamma_df: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    """
    Validation risk-return frontier:
        x = mean average squared inventory
        y = validation mean final PV
    """
    df = gamma_df.copy()
    df["selected_bool"] = selected_bool(df["selected"])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(df["mean_avg_q2"], df["mean_final_pv"], s=80)

    annotate_points(
        ax,
        df["mean_avg_q2"],
        df["mean_final_pv"],
        df["gamma_label"],
    )

    selected = df[df["selected_bool"]]
    if not selected.empty:
        ax.scatter(
            selected["mean_avg_q2"],
            selected["mean_final_pv"],
            s=180,
            marker="*",
        )
        for _, row in selected.iterrows():
            ax.annotate(
                "selected",
                (row["mean_avg_q2"], row["mean_final_pv"]),
                xytext=(8, -14),
                textcoords="offset points",
                fontsize=9,
            )

    threshold = float(df["inventory_risk_threshold"].dropna().iloc[0]) if "inventory_risk_threshold" in df else np.nan
    x_max = float(df["mean_avg_q2"].max())
    if np.isfinite(threshold) and threshold <= 1.5 * x_max:
        ax.axvline(threshold, linestyle="--", linewidth=1)
        ax.annotate("inventory-risk threshold", (threshold, df["mean_final_pv"].min()), xytext=(5, 5), textcoords="offset points")

    ax.set_title("AS validation gamma selection: risk-return frontier")
    ax.set_xlabel("Validation mean average squared inventory")
    ax.set_ylabel("Validation mean final portfolio value")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, out_dir, "01_as_gamma_selection_frontier", formats, dpi)


def plot_gamma_selection_by_gamma(
    gamma_df: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    """
    Validation metrics as a function of gamma.
    Uses a single axis by normalising PV and Q² to their own ranges.
    This avoids subplots while still showing the trade-off shape.
    """
    df = gamma_df.copy()
    df["selected_bool"] = selected_bool(df["selected"])

    # Keep labels readable. Use ordinal x positions but annotate with gamma values in table/labels.
    df = df.reset_index(drop=True)
    x = np.arange(len(df))

    def minmax(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        lo, hi = float(s.min()), float(s.max())
        if math.isclose(lo, hi):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - lo) / (hi - lo)

    pv_norm = minmax(df["mean_final_pv"])
    q2_norm = minmax(df["mean_avg_q2"])
    std_norm = minmax(df["std_final_pv"])

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.plot(x, pv_norm, marker="o", label="mean final PV, normalised")
    ax.plot(x, q2_norm, marker="o", label="avg Q², normalised")
    ax.plot(x, std_norm, marker="o", label="std final PV, normalised")

    selected_idx = df.index[df["selected_bool"]].to_list()
    for idx in selected_idx:
        ax.axvline(idx, linestyle="--", linewidth=1)
        ax.annotate("selected", (idx, 1.0), xytext=(5, -15), textcoords="offset points", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["gamma_label"], rotation=20, ha="right")
    ax.set_title("AS validation gamma sweep: normalised metrics")
    ax.set_xlabel("Gamma setting")
    ax.set_ylabel("Normalised value")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, out_dir, "02_as_gamma_selection_by_gamma", formats, dpi)


def plot_test_risk_return_frontier(
    strategy_summary: pd.DataFrame,
    daily_summary: pd.DataFrame | None,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    """
    Test risk-return plot:
        x = mean average squared inventory
        y = mean final PV
    Uses daily CI if available.
    """
    summary = strategy_summary.copy()

    if daily_summary is not None and not daily_summary.empty:
        ci = daily_summary[["strategy", "daily_ci95_final_pv", "daily_ci95_avg_q2"]]
        summary = summary.merge(ci, on="strategy", how="left")
    else:
        summary["daily_ci95_final_pv"] = np.nan
        summary["daily_ci95_avg_q2"] = np.nan

    order = clean_strategy_order(summary["strategy"])
    summary["order"] = summary["strategy"].map({s: i for i, s in enumerate(order)})
    summary = summary.sort_values("order")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    x = summary["mean_avg_q2"].astype(float)
    y = summary["mean_final_pv"].astype(float)
    xerr = summary["daily_ci95_avg_q2"].astype(float)
    yerr = summary["daily_ci95_final_pv"].astype(float)

    # Only use error bars where nonzero and finite.
    xerr = xerr.where(np.isfinite(xerr), np.nan)
    yerr = yerr.where(np.isfinite(yerr), np.nan)

    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        markersize=7,
        capsize=3,
        linestyle="none",
    )

    annotate_points(ax, x, y, summary["strategy_label"])

    ax.axhline(0.0, linewidth=1)
    ax.set_title("AS test benchmark: risk-return frontier")
    ax.set_xlabel("Mean average squared inventory")
    ax.set_ylabel("Mean final portfolio value")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, out_dir, "03_as_test_risk_return_frontier", formats, dpi)


def plot_activity_rates(
    strategy_summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    summary = strategy_summary.copy()
    order = clean_strategy_order(summary["strategy"])
    summary["order"] = summary["strategy"].map({s: i for i, s in enumerate(order)})
    summary = summary.sort_values("order").set_index("strategy_label")

    rate_cols = [
        "mean_order_submission_rate",
        "mean_fill_rate",
        "mean_no_trade_rate",
    ]
    labels = {
        "mean_order_submission_rate": "submission",
        "mean_fill_rate": "fill",
        "mean_no_trade_rate": "no-trade",
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    summary[rate_cols].rename(columns=labels).plot(kind="bar", ax=ax)

    ax.set_title("AS test benchmark: activity and realised fills")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    ax.legend(title="")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, out_dir, "04_as_test_activity_rates", formats, dpi)


def plot_daily_mean_final_pv(
    daily: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    order = clean_strategy_order(df["strategy"])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for strategy in order:
        part = df[df["strategy"].eq(strategy)].sort_values("date")
        if part.empty:
            continue
        ax.plot(part["date"], part["mean_final_pv"], marker="o", label=label_strategy(strategy))

    as_part = df[df["strategy"].eq("as_classical_control")]
    if not as_part.empty:
        worst = as_part.loc[as_part["mean_final_pv"].idxmin()]
        ax.annotate(
            f"worst AS day\n{worst['date'].date()}",
            (worst["date"], worst["mean_final_pv"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_title("AS test benchmark: daily mean final portfolio value")
    ax.set_xlabel("Test date")
    ax.set_ylabel("Daily mean final PV")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    return save_figure(fig, out_dir, "05_as_daily_mean_final_pv", formats, dpi)


def plot_daily_avg_q2(
    daily: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    order = clean_strategy_order(df["strategy"])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for strategy in order:
        part = df[df["strategy"].eq(strategy)].sort_values("date")
        if part.empty:
            continue
        ax.plot(part["date"], part["mean_avg_q2"], marker="o", label=label_strategy(strategy))

    as_part = df[df["strategy"].eq("as_classical_control")]
    if not as_part.empty:
        worst = as_part.loc[as_part["mean_avg_q2"].idxmax()]
        ax.annotate(
            f"highest AS Q²\n{worst['date'].date()}",
            (worst["date"], worst["mean_avg_q2"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("AS test benchmark: daily average squared inventory")
    ax.set_xlabel("Test date")
    ax.set_ylabel("Daily mean average squared inventory")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()
    return save_figure(fig, out_dir, "06_as_daily_avg_q2", formats, dpi)


def plot_episode_final_pv_distribution(
    metrics: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    central_quantile: float,
) -> list[str]:
    """
    Histogram of episode-level final PV. No-trade is a degenerate zero distribution,
    so it is shown as a vertical reference line.
    """
    df = metrics.copy()
    active = df[~df["strategy"].eq("no_trade")].copy()
    if active.empty:
        active = df.copy()

    values = active["final_pv"].astype(float)
    lo = float(values.quantile((1.0 - central_quantile) / 2.0))
    hi = float(values.quantile(1.0 - (1.0 - central_quantile) / 2.0))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    order = clean_strategy_order(df["strategy"])
    for strategy in order:
        part = df[df["strategy"].eq(strategy)]
        if part.empty:
            continue
        if strategy == "no_trade":
            ax.axvline(0.0, linestyle="--", linewidth=1, label="No-trade PV = 0")
            continue

        pv = part["final_pv"].astype(float)
        pv = pv[(pv >= lo) & (pv <= hi)]
        ax.hist(
            pv,
            bins=80,
            alpha=0.45,
            density=True,
            label=label_strategy(strategy),
        )

    ax.axvline(0.0, linewidth=1)
    ax.set_title(f"AS test benchmark: episode final PV distribution, central {central_quantile:.0%}")
    ax.set_xlabel("Episode final portfolio value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, out_dir, "07_as_episode_final_pv_distribution", formats, dpi)


def plot_daily_fill_vs_pv(
    daily: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    """
    Daily fill rate vs daily mean final PV.
    Marker size scales with daily mean average Q².
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    order = clean_strategy_order(df["strategy"])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for strategy in order:
        part = df[df["strategy"].eq(strategy)].copy()
        if part.empty:
            continue

        q2 = part["mean_avg_q2"].astype(float)
        if float(q2.max()) > 0:
            sizes = 40.0 + 180.0 * q2 / float(q2.max())
        else:
            sizes = pd.Series(np.full(len(part), 40.0), index=part.index)

        ax.scatter(
            part["mean_fill_rate"],
            part["mean_final_pv"],
            s=sizes,
            alpha=0.75,
            label=label_strategy(strategy),
        )

    as_part = df[df["strategy"].eq("as_classical_control")]
    if not as_part.empty:
        # Annotate the most negative PV day and highest Q² day.
        idxs = {as_part["mean_final_pv"].idxmin(), as_part["mean_avg_q2"].idxmax()}
        for idx in idxs:
            row = as_part.loc[idx]
            ax.annotate(
                str(row["date"].date()),
                (row["mean_fill_rate"], row["mean_final_pv"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )

    ax.axhline(0.0, linewidth=1)
    ax.set_title("AS test benchmark: daily fill rate versus daily final PV")
    ax.set_xlabel("Daily mean fill rate")
    ax.set_ylabel("Daily mean final PV")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, out_dir, "08_as_daily_fill_vs_pv", formats, dpi)


def plot_episode_q2_vs_pv(
    metrics: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    max_points: int,
    random_seed: int,
) -> list[str]:
    """
    Episode-level risk-return cloud. Useful for seeing whether high inventory-risk
    episodes correspond to worse PV.
    """
    df = metrics.copy()
    df = df[~df["strategy"].eq("no_trade")].copy()

    if df.empty:
        warnings.warn("No non-no-trade rows available for episode Q² vs PV plot.")
        return []

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=random_seed)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    order = clean_strategy_order(df["strategy"])
    for strategy in order:
        part = df[df["strategy"].eq(strategy)]
        if part.empty:
            continue
        ax.scatter(
            part["avg_q2"],
            part["final_pv"],
            s=14,
            alpha=0.25,
            label=label_strategy(strategy),
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_title("AS test benchmark: episode PV versus average squared inventory")
    ax.set_xlabel("Episode average squared inventory")
    ax.set_ylabel("Episode final PV")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, out_dir, "09_as_episode_q2_vs_pv", formats, dpi)


def plot_episode_drawdown_distribution(
    metrics: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    central_quantile: float,
) -> list[str]:
    """
    Episode drawdown distribution. The raw max_drawdown is usually <= 0.
    We plot drawdown magnitude = -min(max_drawdown, 0).
    """
    df = metrics.copy()
    df["drawdown_magnitude"] = -df["max_drawdown"].astype(float).clip(upper=0.0)

    active = df[~df["strategy"].eq("no_trade")].copy()
    if active.empty:
        active = df.copy()

    values = active["drawdown_magnitude"].astype(float)
    lo = float(values.quantile((1.0 - central_quantile) / 2.0))
    hi = float(values.quantile(1.0 - (1.0 - central_quantile) / 2.0))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    order = clean_strategy_order(df["strategy"])
    for strategy in order:
        part = df[df["strategy"].eq(strategy)].copy()
        if part.empty:
            continue
        if strategy == "no_trade":
            ax.axvline(0.0, linestyle="--", linewidth=1, label="No-trade drawdown = 0")
            continue

        dd = part["drawdown_magnitude"].astype(float)
        dd = dd[(dd >= lo) & (dd <= hi)]
        ax.hist(
            dd,
            bins=80,
            alpha=0.45,
            density=True,
            label=label_strategy(strategy),
        )

    ax.set_title(f"AS test benchmark: episode drawdown distribution, central {central_quantile:.0%}")
    ax.set_xlabel("Episode drawdown magnitude")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    return save_figure(fig, out_dir, "10_as_episode_drawdown_distribution", formats, dpi)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AS thesis plots from results/AS.")
    parser.add_argument("--as-dir", type=Path, default=Path("results/AS"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/AS/thesis_plots"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--central-quantile", type=float, default=0.98, help="Central fraction for clipped distribution plots.")
    parser.add_argument("--max-scatter-points", type=int, default=25000)
    parser.add_argument("--random-seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    as_dir: Path = args.as_dir
    baseline_dir: Path = args.baseline_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gamma_path = as_dir / "as_gamma_selection.csv"
    metrics_path = as_dir / "as_test_metrics.csv"
    daily_path = as_dir / "as_test_daily_metrics.csv"
    summary_path = as_dir / "as_test_summary.json"
    no_trade_path = baseline_dir / "no_trade_for_rl_comparison.csv"

    gamma_df = read_csv_required(gamma_path)
    metrics = read_csv_required(metrics_path)
    daily = read_csv_required(daily_path)
    summary_json = read_json_required(summary_path)

    gamma_df = coerce_numeric(gamma_df, skip={"gamma_label", "selected"})
    metrics = coerce_numeric(
        metrics,
        skip={
            "strategy",
            "baseline_family",
            "split",
            "date",
            "episode_id",
            "data_format",
            "fill_model",
            "horizon_mode",
        },
    )
    daily = coerce_numeric(daily, skip={"date", "strategy"})

    # If no_trade is missing from metrics, append optional no-trade file.
    if "strategy" in metrics.columns and "no_trade" not in set(metrics["strategy"].astype(str)):
        if no_trade_path.exists() and no_trade_path.stat().st_size > 0:
            no_trade = pd.read_csv(no_trade_path)
            no_trade = coerce_numeric(
                no_trade,
                skip={
                    "strategy",
                    "baseline_family",
                    "split",
                    "date",
                    "episode_id",
                    "data_format",
                    "fill_model",
                    "horizon_mode",
                },
            )
            metrics = pd.concat([metrics, no_trade], ignore_index=True)
        else:
            warnings.warn("No no_trade rows found in as_test_metrics.csv and optional no_trade file is missing.")

    # Clean date columns.
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    metrics["date"] = pd.to_datetime(metrics["date"]).dt.date

    # Aggregated summaries for tables and uncertainty estimates.
    strategy_summary = episode_strategy_summary(metrics)
    daily_summary = daily_ci_summary(daily)

    strategy_summary.to_csv(out_dir / "as_test_strategy_summary.csv", index=False)
    daily_summary.to_csv(out_dir / "as_daily_strategy_summary.csv", index=False)

    written: list[str] = []

    written.extend(plot_gamma_selection_frontier(gamma_df, out_dir, args.formats, args.dpi))
    written.extend(plot_gamma_selection_by_gamma(gamma_df, out_dir, args.formats, args.dpi))
    written.extend(plot_test_risk_return_frontier(strategy_summary, daily_summary, out_dir, args.formats, args.dpi))
    written.extend(plot_activity_rates(strategy_summary, out_dir, args.formats, args.dpi))
    written.extend(plot_daily_mean_final_pv(daily, out_dir, args.formats, args.dpi))
    written.extend(plot_daily_avg_q2(daily, out_dir, args.formats, args.dpi))
    written.extend(plot_episode_final_pv_distribution(metrics, out_dir, args.formats, args.dpi, args.central_quantile))
    written.extend(plot_daily_fill_vs_pv(daily, out_dir, args.formats, args.dpi))
    written.extend(plot_episode_q2_vs_pv(metrics, out_dir, args.formats, args.dpi, args.max_scatter_points, args.random_seed))
    written.extend(plot_episode_drawdown_distribution(metrics, out_dir, args.formats, args.dpi, args.central_quantile))

    manifest = {
        "inputs": {
            "as_gamma_selection": str(gamma_path),
            "as_test_metrics": str(metrics_path),
            "as_test_daily_metrics": str(daily_path),
            "as_test_summary": str(summary_path),
            "optional_no_trade_for_rl_comparison": str(no_trade_path),
        },
        "outputs": written,
        "tables": [
            str(out_dir / "as_test_strategy_summary.csv"),
            str(out_dir / "as_daily_strategy_summary.csv"),
        ],
        "summary_json_metadata": summary_json.get("metadata", {}),
        "strategies_in_metrics": sorted(metrics["strategy"].astype(str).unique().tolist()),
        "notes": [
            "Use 01 and 02 for validation gamma selection.",
            "Use 03 and 04 as the compact AS benchmark plots.",
            "Use 05 and 06 for daily robustness and stress-day discussion.",
            "Use 07, 08, 09, and 10 as distribution/activity diagnostics.",
            "If symmetric_mm is absent, rerun AS with --include-symmetric before final thesis comparison.",
        ],
    }

    with (out_dir / "plot_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(written)} figure files to {out_dir}")
    print(f"Wrote summary tables to {out_dir}")
    print("Generated files:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
