"""Kernel-level validation of the compact H2 algorithms against the full (dense) reference, and the
invariant that the full H2 is never built. Data: $AFQMC_DATA/test/diamond_kpoint (skipped if missing)."""
import os
import sys
import tracemalloc
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import functions as new
from opt_einsum import contract
from test_compact_hamiltonian import both, make_config, perturbed, rel, DENSE, COMPACT, NK, NB, NE, NG, DTAU

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(DENSE) and os.path.isdir(COMPACT)),
    reason="k-point test data not found (set AFQMC_DATA)")


def stats(a, b):
    """(max abs, rms, relative to max|b|) of a - b."""
    d = np.abs(np.asarray(a) - np.asarray(b))
    return d.max(), np.sqrt(np.mean(d**2)), d.max() / max(np.abs(np.asarray(b)).max(), 1e-300)


@pytest.fixture(scope="module")
def system():
    config = make_config(6)
    Hd, Hc, trial, walkers = both(config)
    walkers = perturbed(config, walkers)
    theta = new.biorthogonalize(config.backend, trial, walkers.slater_det)
    return config, Hd, Hc, trial, walkers, theta


def test_mean_field_pieces_match_dense(system):
    """L_0, H_zero, the mean-field one-body change and h_sic, each against the dense helpers."""
    config, Hd, Hc, trial, walkers, theta = system
    L = new.compact_to_dense(np.load(os.path.join(COMPACT, "H2_zip.npy")),
                             new.build_kmap(np.load(os.path.join(COMPACT, "Q_list.npy")).T, NK),
                             np.load(os.path.join(COMPACT, "Q_sizes.npy")), NB)           # reference only
    Ld = L
    dag = np.einsum("prG->rpG", Ld.conj())
    ql = new.obtain_Q_list(config, os.path.join(COMPACT, "Q_list.npy"))
    h1 = np.asarray(Hd.one_body)
    trial_single = np.asarray(trial)[:NB, :NE]
    h_mf = new.H_1_mf(trial_single, np.asarray(trial), Ld, dag, ql, h1, NK, NB, NE)
    assert stats(Hc._change / 2, h_mf - h1)[2] < 1e-12                                     # mean-field one-body change
    h_sic = -contract("ijG, jkG -> ik", Ld, dag) / (2 * NK)
    assert stats(Hc._h_sic, h_sic)[2] < 1e-12
    L0 = new.mean_field_diag(Ld, NE, NB, NK)
    assert stats(Hc._L0, L0)[2] < 1e-12
    assert abs(Hc.H_zero - 2 * np.einsum("g,g->", L0, L0.conj()) / (2 * NE * NK**2)) < 1e-12 * abs(Hc.H_zero)
    assert abs(Hc.H_zero - Hd.H_zero) < 1e-12 * abs(Hd.H_zero)


def test_force_bias_and_pq_match_dense(system):
    config, Hd, Hc, trial, walkers, theta = system
    assert stats(Hc._force_bias(theta), Hd._force_bias_expression(theta))[2] < 1e-12


def test_auxiliary_field_matrix_matches_dense(system):
    config, Hd, Hc, trial, walkers, theta = system
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2 * NG, 6)) + 0.3j * rng.standard_normal((2 * NG, 6))
    xb = config.backend.array(x, dtype=config.complex_type)
    f_c, f_d = Hc._auxiliary_field(xb), Hd._auxiliary_field(xb)
    assert np.asarray(f_c).shape == (NB * NK, NB * NK, 6)
    assert stats(f_c, f_d)[2] < 1e-12


def test_energy_components_match_dense(system):
    config, Hd, Hc, trial, walkers, theta = system
    assert stats(Hc.compute_one_body(theta), Hd.compute_one_body(theta))[2] < 1e-13
    assert stats(Hc.compute_hartree(theta), Hd.compute_hartree(theta))[2] < 1e-12
    assert stats(Hc.compute_exchange(theta), Hd.compute_exchange(theta))[2] < 1e-12
    e_c = new.measure_energy(config, trial, walkers, Hc)[0]
    e_d = new.measure_energy(config, trial, walkers, Hd)[0]
    assert abs(e_c - e_d) < 1e-9


def test_exchange_gram_and_direct_agree(system):
    """The per-sector Gram form and the independent sum-over-g form give the same exchange."""
    config, Hd, Hc, trial, walkers, theta = system
    Hc.exchange_mode = "gram"
    gram = np.asarray(Hc.compute_exchange(theta))
    Hc.exchange_mode = "direct"
    try:
        direct = np.asarray(Hc.compute_exchange(theta))
    finally:
        Hc.exchange_mode = "gram"
    assert stats(gram, direct)[2] < 1e-12
    assert stats(direct, Hd.compute_exchange(theta))[2] < 1e-12


def test_hartree_matches_dense_for_hf_trial_walkers():
    config = make_config(3)
    Hd, Hc, trial, walkers = both(config)
    th = new.biorthogonalize(config.backend, trial, walkers.slater_det)
    assert stats(Hc.compute_hartree(th), Hd.compute_hartree(th))[2] < 1e-13
    assert stats(Hc.compute_exchange(th), Hd.compute_exchange(th))[2] < 1e-13


def test_qr_and_weight_update_match_dense(system):
    config, Hd, Hc, trial, walkers, theta = system
    field = config.backend.array(np.random.default_rng(21).standard_normal((2 * NG, 6)), dtype=config.float_type)
    Hd.test_random_field = Hc.test_random_field = field
    e0 = new.measure_energy(config, trial, walkers, Hd)[0]
    nd, _ = new.propagate_walkers(config, trial, walkers, Hd, np.exp(DTAU * Hd.H_zero), e0)
    nc, _ = new.propagate_walkers(config, trial, walkers, Hc, np.exp(DTAU * Hc.H_zero), e0)
    assert stats(nc.weights, nd.weights)[2] < 1e-12
    qd, qc = new.reortho_qr(config, nd.slater_det), new.reortho_qr(config, nc.slater_det)
    assert stats(qc, qd)[2] < 1e-11


# ---------------------------------------------------------------------------------------------------------
#  invariant: the full (nb*nk, nb*nk, ng) H2 is never built by the compact path
# ---------------------------------------------------------------------------------------------------------
def test_compact_path_never_calls_dense_reconstruction(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dense H2 reconstruction called in the compact path")
    monkeypatch.setattr(new, "compact_to_dense", boom)
    config = make_config(4)
    q = new.obtain_Q_list(config, os.path.join(COMPACT, "Q_list.npy"))
    h2c, layout = new.obtain_H2_host(config, os.path.join(COMPACT, "H2_zip.npy"))
    assert layout == "compact"
    H = new.build_hamiltonian(config, new.obtain_H1(config, os.path.join(COMPACT, "H1_svd.npy")), h2c, layout, q,
                              new.obtain_Q_sizes(config, os.path.join(COMPACT, "Q_sizes.npy")))
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    assert type(H) is new.HamiltonianCompact and H.two_body is None
    e0 = new.measure_energy(config, trial, walkers, H)[0]
    for _ in range(2):
        walkers, _ = new.propagate_walkers(config, trial, walkers, H, np.exp(DTAU * H.H_zero), e0)
        walkers.slater_det = new.reortho_qr(config, walkers.slater_det)
        e0 = new.measure_energy(config, trial, walkers, H)[0]


def test_no_array_with_the_size_of_the_full_h2():
    """No stored array of the compact Hamiltonian approaches the size of the full H2, and the host memory
    allocated during setup plus a few steps stays well below one full-H2 tensor."""
    config = make_config(4)
    q = new.obtain_Q_list(config, os.path.join(COMPACT, "Q_list.npy"))
    h1 = new.obtain_H1(config, os.path.join(COMPACT, "H1_svd.npy"))
    sizes = new.obtain_Q_sizes(config, os.path.join(COMPACT, "Q_sizes.npy"))
    h2c, layout = new.obtain_H2_host(config, os.path.join(COMPACT, "H2_zip.npy"))
    full_bytes = NB * NK * NB * NK * NG * 16                                   # complex128 full H2
    tracemalloc.start()
    H = new.build_hamiltonian(config, h1, h2c, layout, q, sizes)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    e0 = new.measure_energy(config, trial, walkers, H)[0]
    for _ in range(2):
        walkers, _ = new.propagate_walkers(config, trial, walkers, H, np.exp(DTAU * H.H_zero), e0)
        e0 = new.measure_energy(config, trial, walkers, H)[0]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.6 * full_bytes, (peak / 2**20, full_bytes / 2**20)
    for name, v in vars(H).items():
        if hasattr(v, "nbytes") and getattr(v, "ndim", 0) > 0:
            assert v.nbytes < 0.2 * full_bytes, (name, v.shape)
            assert not (v.ndim >= 3 and v.shape[0] == NB * NK and v.shape[1] == NB * NK), (name, v.shape)
