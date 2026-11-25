"""Gamma point with real vectors (REALG = Auto/On) against the complex vectors (REALG = Off).

The real path writes the same V with n real symmetric vectors and n real fields; the complex path uses 2n fields
(the Hermitian pair (L+L^+)/2, i(L-L^+)/2 of every complex vector). The energies are identical; the fields differ
by an orthogonal map of the Gaussian variables, so the quantities that do not depend on the basis of the fields
are compared: the covariance of the field operator sum_k A_k (x) A_k, the force-bias operator sum_k f_k A_k and
sum_k f_k^2.
Data: $AFQMC_DATA/test/diamond_kpoint/supercell_gamma (skipped if missing).
"""
import os
import sys
import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(__file__))
import functions as new
import timing

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get(
    "AFQMC_DATA", os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "data")))
SUPER = os.path.join(_DATA_ROOT, "test", "diamond_kpoint", "supercell_gamma")
pytestmark = pytest.mark.skipif(not os.path.isdir(SUPER), reason="Gamma-point test data not found (set AFQMC_DATA)")

NB, NE, NG, DTAU = 64, 32, 1769, 2.5e-4


def make_config(nw, real_gamma, precision="Double", seed=5, backend=None):
    return new.Configuration(
        num_walkers=nw, num_kpoint=1, num_orbital=NB, num_electron=NE, num_g=NG, singularity=0.0,
        propagator="S2", order_propagation=6, timestep=DTAU, comm=MPI.COMM_WORLD, precision=precision,
        backend=backend or new.NumpyBackend(seed), real_gamma=real_gamma)


def hamiltonian(config, h2=None):
    h1 = new.obtain_H1(config, os.path.join(SUPER, "H1_svd.npy"))
    if h2 is None:
        h2 = np.load(os.path.join(SUPER, "H2_zip.npy")).astype(config.complex_type)
    H = new.build_hamiltonian(config, h1, h2, "dense", np.ones((1, 3), dtype=np.int64))
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    return H, trial, walkers


def perturbed(config, walkers, seed=3, amp=0.02):
    rng = np.random.default_rng(seed)
    sd = np.asarray(walkers.slater_det)
    sd = sd + amp * (rng.standard_normal(sd.shape) + 1j * rng.standard_normal(sd.shape))
    walkers.slater_det = config.backend.array(sd, dtype=config.complex_type)
    return walkers


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def operators(H, nf):
    """Field operator of every unit field (nf, nb, nb), mean field included."""
    return np.asarray(H._chans[0].k["aux"](H._chans[0].M, H._c, H._c_conj, H._eye, np.eye(nf, dtype=complex)))


@pytest.fixture(scope="module")
def pair():
    cr, cc = make_config(3, "Auto"), make_config(3, "Off")
    Hr, trial, walkers = hamiltonian(cr)
    Hc, _, _ = hamiltonian(cc)
    walkers = perturbed(cr, walkers)
    th = new.biorthogonalize(cr.backend, trial, walkers.slater_det)
    return cr, Hr, Hc, trial, walkers, th


def test_auto_selects_real_vectors_with_n_fields(pair):
    _, Hr, Hc, _, _, _ = pair
    assert Hr.real_gamma and not Hc.real_gamma
    assert Hr.num_fields == NG and Hc.num_fields == 2 * NG
    assert Hr._M.dtype == np.float64 and Hr._G.dtype == np.float64
    assert Hr.real_gamma_info["imag_ratio"] < 1e-10 and Hr.real_gamma_info["asym"] < 1e-8


def test_energies_and_mean_field_match_complex(pair):
    _, Hr, Hc, _, _, th = pair
    assert abs(Hr.H_zero - Hc.H_zero) < 1e-12 * abs(Hc.H_zero)
    assert rel(Hr.exp_h1_half, Hc.exp_h1_half) < 1e-12
    assert rel(Hr.compute_one_body(th), Hc.compute_one_body(th)) < 1e-13
    assert rel(Hr.compute_hartree(th), Hc.compute_hartree(th)) < 1e-11
    assert rel(Hr.compute_exchange(th), Hc.compute_exchange(th)) < 1e-11


def test_field_covariance_and_force_bias_match_complex(pair):
    _, Hr, Hc, _, _, th = pair
    Ar, Ac = operators(Hr, Hr.num_fields), operators(Hc, Hc.num_fields)
    cov_r = np.einsum("kij,klm->ijlm", Ar, Ar, optimize=True)
    cov_c = np.einsum("kij,klm->ijlm", Ac, Ac, optimize=True)
    assert rel(cov_r, cov_c) < 1e-10
    fr, fc = np.asarray(Hr._force_bias(th)), np.asarray(Hc._force_bias(th))
    assert rel((fr ** 2).sum(axis=0), (fc ** 2).sum(axis=0)) < 1e-10
    assert rel(np.einsum("kw,kij->wij", fr, Ar), np.einsum("kw,kij->wij", fc, Ac)) < 1e-10


def test_propagation_step_with_mapped_field_matches_complex(pair):
    """x_c = T^T y maps n real fields onto the 2n complex ones (A_c = T^T A_r): same step."""
    cr, Hr, Hc, trial, walkers, _ = pair
    Ar, Ac = operators(Hr, Hr.num_fields), operators(Hc, Hc.num_fields)
    T, *_ = np.linalg.lstsq(Ar.reshape(NG, -1).T, Ac.reshape(2 * NG, -1).T, rcond=None)   # A_c[k] = sum_g T[g,k] A_r[g]
    assert rel(np.einsum("gk,gij->kij", T, Ar), Ac) < 1e-10
    assert rel(T @ T.T, np.eye(NG)) < 1e-8
    y = np.random.default_rng(2).standard_normal((NG, 3))
    Hr.test_random_field = y
    Hc.test_random_field = T.T @ y + (np.eye(2 * NG) - T.T @ T) @ np.random.default_rng(4).standard_normal((2 * NG, 3))
    out = []
    for H in (Hr, Hc):      # own walkers: the energy keeps P, Q (in the basis of its fields) for the next force bias
        w = perturbed(cr, new.initialize_determinant(cr)[1])
        e0 = new.measure_energy(cr, trial, w, H)[0]
        out.append(new.propagate_walkers(cr, trial, w, H, np.exp(DTAU * H.H_zero), e0)[0])
    nr, nc = out
    Hr.test_random_field = Hc.test_random_field = None
    assert rel(nr.slater_det, nc.slater_det) < 1e-10
    assert rel(nr.weights, nc.weights) < 1e-10


def test_complex_v_falls_back_or_raises():
    rng = np.random.default_rng(0)
    h2 = (np.load(os.path.join(SUPER, "H2_zip.npy")) * np.exp(1j * rng.uniform(0, 1, (NB, NB, 1)))).astype(complex)
    H, _, _ = hamiltonian(make_config(2, "Auto"), h2)
    assert not H.real_gamma and H.num_fields == 2 * NG and "reason" in H.real_gamma_info
    with pytest.raises(ValueError):
        hamiltonian(make_config(2, "On"), h2)


def test_realg_on_needs_single_k():
    kdir = os.path.join(os.path.dirname(SUPER), "primitive_k2x2x2_compact")
    if not os.path.isdir(kdir):
        pytest.skip("k-point test data not found")
    config = new.Configuration(
        num_walkers=2, num_kpoint=8, num_orbital=8, num_electron=4, num_g=1999, singularity=0.0, propagator="S2",
        order_propagation=6, timestep=DTAU, comm=MPI.COMM_WORLD, precision="Double", backend=new.NumpyBackend(5),
        real_gamma="On")
    h1 = new.obtain_H1(config, os.path.join(os.path.dirname(SUPER), "primitive_k2x2x2", "H1_svd.npy"))
    q = new.obtain_Q_list(config, os.path.join(os.path.dirname(SUPER), "primitive_k2x2x2", "Q_list.npy"))
    h2, layout = new.obtain_H2_host(config, os.path.join(kdir, "H2_zip.npy"))
    H = new.build_hamiltonian(config, h1, h2, layout, q, new.obtain_Q_sizes(config, os.path.join(kdir, "Q_sizes.npy")))
    with pytest.raises(ValueError, match="KPOINT = 1"):
        H.setup_energy_expressions(config, new.initialize_determinant(config)[0])


def test_jax_real_path_matches_numpy(pair):
    jax = pytest.importorskip("jax")  # noqa: F841
    cr, Hr, _, trial, walkers, th = pair
    cj = make_config(3, "Auto", backend=new.JaxBackend(5, double=True))
    Hj, trial_j, _ = hamiltonian(cj)
    thj = cj.backend.array(np.asarray(th))
    assert Hj.real_gamma
    assert rel(Hj.compute_hartree(thj), Hr.compute_hartree(th)) < 1e-12
    assert rel(Hj.compute_exchange(thj), Hr.compute_exchange(th)) < 1e-12
    assert rel(Hj._force_bias(thj), Hr._force_bias(th)) < 1e-12
    x = np.random.default_rng(1).standard_normal((NG, 3)) + 0.1j
    assert rel(Hj._auxiliary_field(cj.backend.array(x)), Hr._auxiliary_field(x)) < 1e-12


def test_operation_counts_real():
    c = timing.operation_counts(1, NB, NE, NG, NG, 10, 6, real=False)
    r = timing.operation_counts(1, NB, NE, NG, NG, 10, 6, real=True)
    assert r["propagate/aux_field/aux_matrix"] == c["propagate/aux_field/aux_matrix"] / 4
    assert r["energy/hartree"] == c["energy/hartree"] / 4
    assert r["energy/exchange"] == c["energy/exchange"] / 2
    assert r["propagate/taylor"] == c["propagate/taylor"]
