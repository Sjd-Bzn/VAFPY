"""Compact H2 on the JAX backend (CPU or GPU, single precision) against NumPy double precision.

Run on a GPU host to exercise the device path; on a CPU-only host JAX falls back to the CPU.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
jax = pytest.importorskip("jax")
import functions as new
from test_compact_hamiltonian import (DENSE, COMPACT, SUPER, NK, NB, NE, NG, DTAU,
                                      make_config, rel, perturbed)
from mpi4py import MPI  # noqa: F401

pytestmark = pytest.mark.skipif(
    not all(os.path.isdir(d) for d in (DENSE, COMPACT, SUPER)),
    reason="k-point test data not found (set AFQMC_DATA)")


def build_pair(backend_name, precision, nw):
    be = new.NumpyBackend(5) if backend_name == "numpy" else new.JaxBackend(5)
    config = make_config(nw, precision=precision, backend=be)
    q = new.obtain_Q_list(config, os.path.join(COMPACT, "Q_list.npy"))
    h1 = new.obtain_H1(config, os.path.join(COMPACT, "H1_svd.npy"))
    h2c, layout = new.obtain_H2_host(config, os.path.join(COMPACT, "H2_zip.npy"))
    assert layout == "compact"
    sizes = new.obtain_Q_sizes(config, os.path.join(COMPACT, "Q_sizes.npy"))
    H = new.build_hamiltonian(config, h1, h2c, layout, q, sizes)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    return config, H, trial, walkers


def to_np(x):
    return np.asarray(x)


def test_jax_single_precision_dtypes():
    config, H, trial, walkers = build_pair("jax", "Single", 4)
    walkers = perturbed(config, walkers)
    th = new.biorthogonalize(config.backend, trial, walkers.slater_det)
    fb = H._force_bias(th)
    field = H._auxiliary_field(fb)
    for name, x in {"H1": H.one_body, "H2c": H.two_body, "padded L": H._Lp, "exp_h1_half": H.exp_h1_half,
                    "walkers": walkers.slater_det, "weights": walkers.weights, "theta": th,
                    "force bias": fb, "aux field": field}.items():
        assert x.dtype == np.complex64, (name, x.dtype)
    assert trial.dtype == np.float32 and H._sqrt_tau.dtype == np.float32


def test_jax_compact_matches_numpy_double():
    cfgj, Hj, trialj, wj = build_pair("jax", "Single", 6)
    cfgn, Hn, trialn, wn = build_pair("numpy", "Double", 6)
    wj, wn = perturbed(cfgj, wj), perturbed(cfgn, wn)
    thj = new.biorthogonalize(cfgj.backend, trialj, wj.slater_det)
    thn = new.biorthogonalize(cfgn.backend, trialn, wn.slater_det)
    assert abs(Hj.H_zero - Hn.H_zero) < 1e-6 * abs(Hn.H_zero)
    for name in ("compute_one_body", "compute_hartree", "compute_exchange"):
        assert rel(to_np(getattr(Hj, name)(thj)), to_np(getattr(Hn, name)(thn))) < 2e-5, name
    assert rel(to_np(Hj._force_bias(thj)), to_np(Hn._force_bias(thn))) < 2e-4
    x = np.random.default_rng(2).standard_normal((2 * NG, 6)) + 0.2j
    assert rel(to_np(Hj._auxiliary_field(cfgj.backend.array(x, dtype=cfgj.complex_type))),
               to_np(Hn._auxiliary_field(cfgn.backend.array(x, dtype=cfgn.complex_type)))) < 2e-4


def test_jax_compact_propagation_matches_numpy_double():
    cfgj, Hj, trialj, wj = build_pair("jax", "Single", 6)
    cfgn, Hn, trialn, wn = build_pair("numpy", "Double", 6)
    wj, wn = perturbed(cfgj, wj), perturbed(cfgn, wn)
    f = np.random.default_rng(11).standard_normal((2 * NG, 6))
    Hj.test_random_field = cfgj.backend.array(f, dtype=cfgj.float_type)
    Hn.test_random_field = cfgn.backend.array(f, dtype=cfgn.float_type)
    e0 = float(new.measure_energy(cfgn, trialn, wn, Hn)[0].real)
    nj, rj = new.propagate_walkers(cfgj, trialj, wj, Hj, np.exp(DTAU * Hj.H_zero), e0)
    nn_, rn = new.propagate_walkers(cfgn, trialn, wn, Hn, np.exp(DTAU * Hn.H_zero), e0)
    assert int(rj) == int(rn) == 0
    assert rel(to_np(nj.slater_det), to_np(nn_.slater_det)) < 5e-5
    assert rel(to_np(nj.weights), to_np(nn_.weights)) < 5e-5
    assert nj.slater_det.dtype == np.complex64 and nj.weights.dtype == np.complex64
