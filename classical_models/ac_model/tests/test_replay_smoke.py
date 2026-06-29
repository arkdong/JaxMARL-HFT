import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ac_benchmark.calibration import estimate_temporary_impact_eta
from ac_benchmark.fast_calibration import estimate_temporary_impact_eta_fast
from ac_benchmark.fast_replay import evaluate_ac_grid_fast
from ac_benchmark.replay import evaluate_ac_grid
from ac_benchmark.schema import EpisodeSpec


def make_book(n=500, levels=3):
    mid = 200.0
    data = {"timestamp": list(range(n)), "best_bid": [199.99] * n, "best_ask": [200.01] * n}
    for lvl in range(1, levels + 1):
        data[f"ask_price_{lvl}"] = [200.00 + 0.01 * lvl] * n
        data[f"bid_price_{lvl}"] = [200.00 - 0.01 * lvl] * n
        data[f"ask_size_{lvl}"] = [1000] * n
        data[f"bid_size_{lvl}"] = [1000] * n
    df = pd.DataFrame(data)
    df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2
    df["spread"] = df["best_ask"] - df["best_bid"]
    return df


def test_evaluate_ac_grid_smoke():
    df = make_book()
    spec = EpisodeSpec(task_size=600, episode_length=64, messages_per_step=1, episode_start_frequency_steps=64, directions="buy")
    metrics, fills, plan = evaluate_ac_grid(df, spec=spec, kappa_T_grid=[0.0, 2.0], depth=3, return_fills=True)
    assert not metrics.empty
    assert metrics["unfinished"].max() == 0
    assert set(metrics["kappa_T"]) == {0.0, 2.0}
    assert fills is not None and not fills.empty


def make_variable_book(n=180, levels=3):
    idx = np.arange(n)
    mid = 200.0 + 0.02 * np.sin(idx / 9.0)
    data = {
        "timestamp": idx,
        "best_bid": mid - 0.01,
        "best_ask": mid + 0.01,
    }
    for lvl in range(1, levels + 1):
        data[f"ask_price_{lvl}"] = mid + 0.01 * lvl
        data[f"bid_price_{lvl}"] = mid - 0.01 * lvl
        data[f"ask_size_{lvl}"] = 25 + ((idx + 5 * lvl) % 30) + 0.75
        data[f"bid_size_{lvl}"] = 30 + ((idx + 7 * lvl) % 35) + 0.5
    df = pd.DataFrame(data)
    df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2
    df["spread"] = df["best_ask"] - df["best_bid"]
    return df


def test_fast_replay_matches_slow_episode_metrics():
    df = make_variable_book()
    spec = EpisodeSpec(
        task_size=180,
        episode_length=6,
        messages_per_step=2,
        episode_start_frequency_steps=3,
        lot_size=10,
        directions="alternating",
    )

    slow_metrics, _, slow_plan = evaluate_ac_grid(
        df,
        spec=spec,
        kappa_T_grid=[0.0, 1.5, 3.0],
        depth=3,
        max_episodes=8,
        return_fills=False,
    )
    fast_metrics, fast_fills, fast_plan = evaluate_ac_grid_fast(
        df,
        spec=spec,
        kappa_T_grid=[0.0, 1.5, 3.0],
        depth=3,
        max_episodes=8,
    )

    assert fast_fills is None
    assert_frame_equal(slow_plan.reset_index(drop=True), fast_plan.reset_index(drop=True), check_dtype=False)

    sort_cols = ["episode_id", "kappa_T"]
    slow_metrics = slow_metrics.sort_values(sort_cols).reset_index(drop=True)
    fast_metrics = fast_metrics.sort_values(sort_cols).reset_index(drop=True)
    assert_frame_equal(slow_metrics, fast_metrics, check_dtype=False, rtol=1e-10, atol=1e-9)


def test_fast_impact_calibration_matches_slow_curve():
    df = make_variable_book(n=90, levels=3)
    slow_eta, slow_curve = estimate_temporary_impact_eta(
        df,
        q_grid=[10, 40, 80],
        depth=3,
        max_rows=None,
    )
    fast_eta, fast_curve = estimate_temporary_impact_eta_fast(
        df,
        q_grid=[10, 40, 80],
        depth=3,
        max_rows=None,
    )

    np.testing.assert_allclose(fast_eta, slow_eta, rtol=1e-12, atol=1e-12)
    assert_frame_equal(slow_curve, fast_curve, check_dtype=False, rtol=1e-12, atol=1e-12)
