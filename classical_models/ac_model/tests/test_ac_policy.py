import numpy as np

from ac_benchmark.ac_policy import ac_holdings, ac_schedule


def test_twap_schedule_sums_and_uniform_lots():
    sched = ac_schedule(task_size=600, n_steps=64, kappa_T=0.0, lot_size=10)
    assert sched.sum() == 600
    assert len(sched) == 64
    assert np.all(sched >= 0)
    assert sched.max() <= 20


def test_front_loaded_schedule_sums_and_front_loads():
    slow = ac_schedule(task_size=600, n_steps=64, kappa_T=0.5, lot_size=10)
    fast = ac_schedule(task_size=600, n_steps=64, kappa_T=4.0, lot_size=10)
    assert slow.sum() == 600
    assert fast.sum() == 600
    assert fast[:10].sum() > slow[:10].sum()


def test_holdings_boundary_values():
    h = ac_holdings(task_size=600, n_steps=64, kappa_T=2.0)
    assert h[0] == 600
    assert h[-1] == 0
    assert np.all(np.diff(h) <= 1e-9)
