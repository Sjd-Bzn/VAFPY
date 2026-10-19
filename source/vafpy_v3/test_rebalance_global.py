"""rebalance_global communicates in the precision of the walkers (complex64 for Single) and gives exactly the same
walkers and weights as the previous implementation with complex128 buffers.

Run with one rank (pytest) or several ranks (mpiexec -n 4 python -m pytest test_rebalance_global.py)."""
import os
import sys
import types

import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(__file__))
import functions as new  # noqa: E402


def reference_rebalance_global(comm, walkers_mats_up, walkers_weights, config):
    """The implementation that converted everything to complex128 (kept as the reference)."""
    walkers_mats_up = np.asarray(walkers_mats_up, dtype=np.complex128)
    walkers_weights = np.asarray(walkers_weights, dtype=np.complex128)
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_n, n_orb, n_elec = walkers_mats_up.shape
    total_n = local_n * size
    all_weights = np.empty(total_n, dtype=np.complex128) if rank == 0 else None
    comm.Gather(walkers_weights, all_weights, root=0)
    instances = None
    if rank == 0:
        norm = total_n / np.sum(all_weights.real)
        norm_w = all_weights.real * norm
        bias = -float(config.backend.random_uniform_scalar())
        instances = np.zeros(total_n, dtype=int)
        cum, prev = bias, 0
        for i in range(total_n):
            cum += norm_w[i]
            cur = int(np.ceil(cum))
            instances[i] = cur - prev
            prev = cur
    if rank != 0:
        instances = np.empty(total_n, dtype=int)
    comm.Bcast(instances, root=0)
    map_indices = np.empty(int(np.sum(instances)), dtype=int)
    count = 0
    for idx, n in enumerate(instances):
        for _ in range(n):
            map_indices[count] = idx
            count += 1
    all_mats = np.empty((total_n, n_orb, n_elec), dtype=np.complex128) if rank == 0 else None
    comm.Gather(walkers_mats_up, all_mats, root=0)
    resampled = all_mats[map_indices] if rank == 0 else None
    new_mats = np.empty((local_n, n_orb, n_elec), dtype=np.complex128)
    comm.Scatter(resampled, new_mats, root=0)
    return new_mats, np.ones(local_n, dtype=np.complex128)


def _walkers(dtype, trial, rank, local_n=50, n_orb=8, n_elec=4):
    rng = np.random.default_rng(1000 * trial + rank)
    mats = (rng.standard_normal((local_n, n_orb, n_elec)) + 1j * rng.standard_normal((local_n, n_orb, n_elec))).astype(dtype)
    w = rng.uniform(0.01, 3.0, local_n)
    if trial % 5 == 0:
        w[rng.integers(0, local_n, 5)] = 1e-6                                      # almost dead walkers
    weights = (w + 1j * 1e-3 * rng.standard_normal(local_n)).astype(dtype)
    return mats, weights


def _config(trial):
    bias = (0.013 + trial * 0.0371) % 1.0                                           # never exactly 0
    return types.SimpleNamespace(backend=types.SimpleNamespace(random_uniform_scalar=lambda: bias))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_rebalance_global_matches_reference(dtype):
    comm = MPI.COMM_WORLD
    for trial in range(30):
        mats, weights = _walkers(dtype, trial, comm.Get_rank())
        m_ref, w_ref = reference_rebalance_global(comm, mats.copy(), weights.copy(), _config(trial))
        m_new, w_new = new.rebalance_global(comm, mats.copy(), weights.copy(), _config(trial))
        assert np.array_equal(m_ref.astype(dtype), m_new)
        assert np.array_equal(w_ref.astype(dtype), w_new)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_rebalance_global_keeps_the_precision(dtype):
    comm = MPI.COMM_WORLD
    mats, weights = _walkers(dtype, 1, comm.Get_rank())
    m_new, w_new = new.rebalance_global(comm, mats, weights, _config(1))
    assert m_new.dtype == dtype and w_new.dtype == dtype
    assert m_new.shape == mats.shape and w_new.shape == weights.shape
