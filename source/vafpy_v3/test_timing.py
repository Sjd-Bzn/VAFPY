"""Timers and the timing report (timing.py)."""
import time

import numpy as np
from mpi4py import MPI

import timing


def test_nested_timers_count_calls_and_time():
    t = timing.Timers(level=2)
    for _ in range(3):
        with t.part("propagate"):
            with t.stage("taylor"):
                time.sleep(0.01)
            with t.stage("exp_h1"):
                pass
            with t.stage("exp_h1"):
                pass
    assert list(t.data) == ["propagate", "propagate/taylor", "propagate/exp_h1"]
    assert t.data["propagate"][0] == 3 and t.data["propagate/exp_h1"][0] == 6
    assert t.data["propagate"][1] >= t.data["propagate/taylor"][1] >= 0.03


def test_levels():
    t1 = timing.Timers(level=1)
    with t1.part("energy"):
        with t1.stage("exchange"):
            pass
    assert list(t1.data) == ["energy"]
    t0 = timing.Timers(level=0)
    with t0.part("energy") as s:
        assert s.done(5) == 5
    assert t0.data == {}


def test_sync_is_called_only_for_done():
    seen = []
    t = timing.Timers(level=2, sync=lambda x: seen.append(x))
    with t.part("reortho") as s:
        s.done("walkers")
    assert seen == ["walkers"]


def test_operation_counts_scale_with_walkers():
    a = timing.operation_counts(8, 8, 4, 1925, 260, 50, 6)
    b = timing.operation_counts(8, 8, 4, 1925, 260, 100, 6)
    for k in a:
        assert a[k] > 0 and np.isclose(b[k], 2 * a[k])
    assert np.isclose(a["propagate/aux_field"], a["propagate/aux_field/force_bias"] + a["propagate/aux_field/aux_matrix"])


def test_report_is_written(tmp_path):
    comm = MPI.COMM_WORLD
    t = timing.Timers(level=2)
    with t.part("setup"):
        pass
    for _ in range(4):
        with t.part("propagate"):
            with t.stage("taylor"):
                pass
    out = tmp_path / f"timing_{comm.Get_rank()}.txt"
    counts = timing.operation_counts(8, 8, 4, 1925, 260, 50, 6)
    lines = timing.write_report(t, comm, str(out), [("backend", "NumPy"), ("reortho", "QR every 1 steps")], counts,
                                n_steps=4)
    if comm.Get_rank() == 0:
        text = out.read_text()
        for key in ("MPI ranks", "BLAS / threads", "Time per part", "propagate", "taylor", "Floating-point operations",
                    "peak resident memory"):
            assert key in text
        assert any(l.startswith("Total run time") for l in lines)
    else:
        assert lines == [] and not out.exists()
