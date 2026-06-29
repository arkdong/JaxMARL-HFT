from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ac_policy import ac_holdings, ac_schedule


def plot_schedules(
    *,
    task_size: int,
    n_steps: int,
    kappa_T_grid: Sequence[float],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(n_steps + 1)
    plt.figure(figsize=(8, 5))
    for kappa_T in kappa_T_grid:
        holdings = ac_holdings(task_size, n_steps, float(kappa_T))
        label = "TWAP" if abs(float(kappa_T)) < 1e-12 else f"κT={float(kappa_T):g}"
        plt.plot(x, holdings, label=label)
    plt.xlabel("Episode step")
    plt.ylabel("Remaining quantity")
    plt.title("Almgren-Chriss remaining-quantity schedules")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_frontier(
    summary_df: pd.DataFrame,
    *,
    out_path: str | Path,
    x_col: str = "avg_sq_remaining_mean",
    y_col: str = "slippage_ticks_per_share_mean",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if x_col not in summary_df.columns or y_col not in summary_df.columns:
        raise ValueError(f"summary_df must contain {x_col} and {y_col}")
    df = summary_df.sort_values("kappa_T")
    plt.figure(figsize=(7, 5))
    plt.plot(df[x_col], df[y_col], marker="o")
    for _, row in df.iterrows():
        plt.annotate(f"κT={row['kappa_T']:g}", (row[x_col], row[y_col]), textcoords="offset points", xytext=(4, 4))
    plt.xlabel("Average squared remaining quantity")
    plt.ylabel("Mean slippage, ticks/share")
    plt.title("AC replay risk-cost frontier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_impact_curve(impact_df: pd.DataFrame, *, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for side, g in impact_df.groupby("side"):
        g = g.sort_values("q")
        plt.plot(g["q"], g["median_impact"], marker="o", label=f"{side} median")
    plt.xlabel("Marketable order size q")
    plt.ylabel("Book-walk impact vs mid price")
    plt.title("Training book-walk impact curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_slippage_distribution(metrics_df: pd.DataFrame, *, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for kappa_T, g in metrics_df.groupby("kappa_T"):
        vals = pd.to_numeric(g["slippage_ticks_per_share"], errors="coerce").dropna().to_numpy()
        if len(vals):
            plt.hist(vals, bins=40, alpha=0.35, label=f"κT={float(kappa_T):g}")
    plt.xlabel("Slippage, ticks/share")
    plt.ylabel("Episode count")
    plt.title("AC slippage distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

