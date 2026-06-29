import pandas as pd

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
