"""A failure on one MPI rank must end the whole job instead of leaving the other ranks waiting."""
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

pytestmark = pytest.mark.skipif(shutil.which("mpiexec") is None, reason="needs mpiexec")


def _run(script, tmp_path, timeout=60):
    path = tmp_path / "driver.py"
    path.write_text(textwrap.dedent(script))
    cmd = ["mpiexec", "-n", "2", sys.executable, str(path)]
    return subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "PYTHONPATH": HERE})


def test_failure_on_one_rank_stops_all_ranks(tmp_path):
    result = _run("""
        import time
        from mpi4py import MPI
        import run_kpts

        def failing_main():
            if MPI.COMM_WORLD.Get_rank() == 1:
                raise ValueError("simulated failure")
            time.sleep(600)          # without the abort, rank 0 would wait here

        run_kpts.main = failing_main
        run_kpts.run()
    """, tmp_path)
    assert result.returncode != 0
    assert "simulated failure" in result.stderr


def test_successful_run_returns_zero(tmp_path):
    result = _run("""
        import run_kpts
        run_kpts.main = lambda: None
        run_kpts.run()
    """, tmp_path)
    assert result.returncode == 0
