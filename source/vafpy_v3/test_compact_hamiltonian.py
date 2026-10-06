"""HamiltonianCompact (compact H2) must reproduce the dense Hamiltonian (full H2) numerically.

Every ingredient of the AFQMC step is compared for the same physical Hamiltonian stored in
both layouts: mean-field setup, local energy, force bias, auxiliary field, a full propagation
step with a fixed random field, and a multi-step trajectory with identical random numbers.
Data: $AFQMC_DATA/test/diamond_kpoint (skipped if missing).
"""
import os
import sys
import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(__file__))
import functions as new

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get(
    "AFQMC_DATA", os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "data")))
KDIR = os.path.join(_DATA_ROOT, "test", "diamond_kpoint")
DENSE, COMPACT, SUPER = (os.path.join(KDIR, d) for d in
                         ("primitive_k2x2x2", "primitive_k2x2x2_compact", "supercell_gamma"))
pytestmark = pytest.mark.skipif(
    not all(os.path.isdir(d) for d in (DENSE, COMPACT, SUPER)),
    reason="k-point test data not found (set AFQMC_DATA)")

DTAU = 2.5e-4
NK, NB, NE, NG = 8, 8, 4, 1999


def make_config(num_walkers, precision="Double", seed=5, nk=NK, nb=NB, ne=NE, ng=NG, backend=None):
    return new.Configuration(
        num_walkers=num_walkers, num_kpoint=nk, num_orbital=nb, num_electron=ne, num_g=ng,
        singularity=0.0, propagator="S2", order_propagation=6, timestep=DTAU,
        comm=MPI.COMM_WORLD, precision=precision, backend=backend or new.NumpyBackend(seed))


def both(config, kmap_dir=DENSE, **kw):
    """(dense Hamiltonian, compact Hamiltonian) for the primitive data, both set up for the HF trial."""
    q = new.obtain_Q_list(config, os.path.join(DENSE, "Q_list.npy"))
    h1 = new.obtain_H1(config, os.path.join(DENSE, "H1_svd.npy"))
    h2d, ld = new.obtain_H2_host(config, os.path.join(DENSE, "H2_zip.npy"))
    h2c, lc = new.obtain_H2_host(config, os.path.join(COMPACT, "H2_zip.npy"))
    assert (ld, lc) == ("dense", "compact")
    sizes = new.obtain_Q_sizes(config, os.path.join(COMPACT, "Q_sizes.npy"))
    Hd = new.build_hamiltonian(config, h1, h2d, ld, q)
    Hc = new.build_hamiltonian(config, h1, h2c, lc, q, sizes)
    trial, walkers = new.initialize_determinant(config)
    Hd.setup_energy_expressions(config, trial)
    Hc.setup_energy_expressions(config, trial)
    return Hd, Hc, trial, walkers


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def perturbed(config, walkers, seed=3, amp=0.05):
    rng = np.random.default_rng(seed)
    sd = np.asarray(walkers.slater_det)
    sd = sd + amp * (rng.standard_normal(sd.shape) + 1j * rng.standard_normal(sd.shape))
    walkers.slater_det = config.backend.array(sd, dtype=config.complex_type)
    return walkers


def test_factory_selects_class_from_layout():
    config = make_config(2)
    Hd, Hc, _, _ = both(config)
    assert type(Hd) is new.Hamiltonian and type(Hc) is new.HamiltonianCompact


def test_mean_field_setup_matches_dense():
    config = make_config(4)
    Hd, Hc, _, _ = both(config)
    assert abs(Hd.H_zero - Hc.H_zero) < 1e-12 * abs(Hd.H_zero)
    assert rel(Hc.h1, Hd.h1) < 1e-12
    assert rel(Hc.exp_h1, Hd.exp_h1) < 1e-12
    assert rel(Hc.exp_h1_half, Hd.exp_h1_half) < 1e-12


def test_local_energy_matches_dense():
    config = make_config(6)
    Hd, Hc, trial, walkers = both(config)
    for w in (walkers, perturbed(config, walkers)):
        th = new.biorthogonalize(config.backend, trial, w.slater_det)
        for name in ("compute_one_body", "compute_hartree", "compute_exchange"):
            assert rel(getattr(Hc, name)(th), getattr(Hd, name)(th)) < 1e-12, name
        e_d = new.measure_energy(config, trial, w, Hd)
        e_c = new.measure_energy(config, trial, w, Hc)
        assert abs(e_d[0] - e_c[0]) < 1e-9


def test_force_bias_matches_dense():
    config = make_config(6)
    Hd, Hc, trial, walkers = both(config)
    walkers = perturbed(config, walkers)
    th = new.biorthogonalize(config.backend, trial, walkers.slater_det)
    fd, fc = Hd._force_bias_expression(th), Hc._force_bias(th)
    assert fc.shape == (2 * NG, 6)
    assert np.abs(np.asarray(fd)).max() > 0.05                # not a trivial comparison
    assert rel(fc, fd) < 1e-12


def test_auxiliary_field_matches_dense():
    config = make_config(5)
    Hd, Hc, _, _ = both(config)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2 * NG, 5)) + 0.3j * rng.standard_normal((2 * NG, 5))   # complex like (random - force bias)
    xb = config.backend.array(x, dtype=config.complex_type)
    fd, fc = Hd._auxiliary_field(xb), Hc._auxiliary_field(xb)
    assert np.asarray(fc).shape == (NB * NK, NB * NK, 5)
    assert rel(fc, fd) < 1e-12


def test_propagation_step_matches_dense():
    config = make_config(6)
    Hd, Hc, trial, walkers = both(config)
    walkers = perturbed(config, walkers)
    field = config.backend.array(np.random.default_rng(11).standard_normal((2 * NG, 6)), dtype=config.float_type)
    Hd.test_random_field = Hc.test_random_field = field
    e0 = new.measure_energy(config, trial, walkers, Hd)[0]
    nd, rd = new.propagate_walkers(config, trial, walkers, Hd, np.exp(DTAU * Hd.H_zero), e0)
    nc, rc = new.propagate_walkers(config, trial, walkers, Hc, np.exp(DTAU * Hc.H_zero), e0)
    assert rd == rc == 0
    assert rel(nc.slater_det, nd.slater_det) < 1e-12
    assert rel(nc.weights, nd.weights) < 1e-12


def _trajectory(which, nsteps=12, nw=8, seed=7):
    config = make_config(nw, seed=seed)
    Hd, Hc, trial, walkers = both(config)
    H = Hd if which == "dense" else Hc
    e0 = new.measure_energy(config, trial, walkers, H)[0]
    h_0 = np.exp(DTAU * H.H_zero)
    energies, weights = [], []
    for j in range(1, nsteps + 1):
        walkers, _ = new.propagate_walkers(config, trial, walkers, H, h_0, e0)
        e0 = new.measure_energy(config, trial, walkers, H)[0]
        energies.append(e0)
        weights.append(np.mean(walkers.weights))
        walkers.slater_det = new.reortho_qr(config, walkers.slater_det)
        if j % 4 == 0:
            idx = new.rebalance_comb(config, walkers.weights)
            walkers.slater_det = walkers.slater_det[idx]
            walkers.weights = new.init_walkers_weights(config, nw)
    return np.array(energies), np.array(weights)


def test_trajectory_with_same_seed_matches_dense():
    """Same seed -> same random fields -> identical walker dynamics (including reorthogonalisation and
    population control) for the dense and the compact Hamiltonian."""
    e_d, w_d = _trajectory("dense")
    e_c, w_c = _trajectory("compact")
    assert np.all(np.isfinite(e_c))
    assert np.max(np.abs(e_c - e_d)) < 1e-8, (e_d, e_c)
    assert np.max(np.abs(w_c - w_d)) < 1e-9


def test_compact_layout_for_single_k_matches_dense():
    """nk=1 supercell: compact (the transposed layout) and dense give the same Hamiltonian."""
    nk, nb, ne, ng = 1, 64, 32, 1769
    config = make_config(3, nk=nk, nb=nb, ne=ne, ng=ng)
    h1 = new.obtain_H1(config, os.path.join(SUPER, "H1_svd.npy"))
    dense = np.load(os.path.join(SUPER, "H2_zip.npy")).astype(np.complex128)       # (64, 64, ng)
    compact = np.ascontiguousarray(dense.transpose(1, 0, 2))                        # Lc[b2, b1, g]
    Hd = new.build_hamiltonian(config, h1, dense, "dense", None)
    Hc = new.HamiltonianCompact(one_body=h1, two_body=config.backend.array(compact, dtype=config.complex_type),
                                kmap=np.zeros((1, 1), dtype=np.int64), sizes=np.array([ng]))
    trial, walkers = new.initialize_determinant(config)
    Hd.setup_energy_expressions(config, trial)
    Hc.setup_energy_expressions(config, trial)
    assert abs(Hd.H_zero - Hc.H_zero) < 1e-10 * abs(Hd.H_zero)
    assert rel(Hc.exp_h1_half, Hd.exp_h1_half) < 1e-11
    walkers = perturbed(config, walkers, amp=0.02)
    th = new.biorthogonalize(config.backend, trial, walkers.slater_det)
    assert rel(Hc.compute_hartree(th), Hd.compute_hartree(th)) < 1e-11
    assert rel(Hc.compute_exchange(th), Hd.compute_exchange(th)) < 1e-11
    assert rel(Hc._force_bias(th), Hd._force_bias_expression(th)) < 1e-10
    field = config.backend.array(np.random.default_rng(1).standard_normal((2 * ng, 3)), dtype=config.float_type)
    Hd.test_random_field = Hc.test_random_field = field
    e0 = new.measure_energy(config, trial, walkers, Hd)[0]
    nd, _ = new.propagate_walkers(config, trial, walkers, Hd, np.exp(DTAU * Hd.H_zero), e0)
    nc, _ = new.propagate_walkers(config, trial, walkers, Hc, np.exp(DTAU * Hc.H_zero), e0)
    assert rel(nc.slater_det, nd.slater_det) < 1e-10


def test_single_precision_compact_stays_single_and_close_to_double():
    cfg32 = make_config(4, precision="Single")
    cfg64 = make_config(4, precision="Double")
    _, H32, trial32, w32 = both(cfg32)
    _, H64, trial64, w64 = both(cfg64)
    assert H32._Lp.dtype == np.complex64 and H32._exp_h1_half.dtype == np.complex64
    w32 = perturbed(cfg32, w32); w64 = perturbed(cfg64, w64)
    th32 = new.biorthogonalize(cfg32.backend, trial32, w32.slater_det)
    th64 = new.biorthogonalize(cfg64.backend, trial64, w64.slater_det)
    fb32 = H32._force_bias(th32)
    assert fb32.dtype == np.complex64 and th32.dtype == np.complex64
    assert rel(fb32, H64._force_bias(th64)) < 5e-5
    assert H32.compute_hartree(th32).dtype in (np.complex64, np.float32)
    assert rel(H32.compute_hartree(th32), H64.compute_hartree(th64)) < 5e-5


def test_compact_hamiltonian_rejects_wrong_inputs():
    config = make_config(2)
    h1 = new.obtain_H1(config, os.path.join(DENSE, "H1_svd.npy"))
    kmap = new.build_kmap(new.obtain_Q_list(config, os.path.join(DENSE, "Q_list.npy")), NK)
    comp = np.load(os.path.join(COMPACT, "H2_zip.npy"))
    sizes = np.load(os.path.join(COMPACT, "Q_sizes.npy"))
    trial, _ = new.initialize_determinant(config)
    bad_sizes = sizes.copy(); bad_sizes[0] += 1
    H = new.HamiltonianCompact(h1, config.backend.array(comp, dtype=config.complex_type), kmap, bad_sizes)
    with pytest.raises(ValueError):
        H.setup_energy_expressions(config, trial)
    H = new.HamiltonianCompact(h1, config.backend.array(comp[:, :, :-1], dtype=config.complex_type), kmap, sizes)
    with pytest.raises(ValueError):
        H.setup_energy_expressions(config, trial)
