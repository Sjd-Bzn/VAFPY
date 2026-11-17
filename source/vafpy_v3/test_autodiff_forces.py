"""Exactness of the derivative of the AFQMC energy by automatic differentiation (autodiff_forces.py), double
precision, JAX on the CPU, two systems:
  gamma: diamond primitive cell, Gamma only, 16 orbitals, 4 occupied, NGVEC 136 (test/diamond_gamma_prim)
  k222:  diamond primitive cell, 2x2x2 k-mesh, 8 bands/k, 4 occupied/k, NGVEC 1999 (test/diamond_kpoint, compact)

  - the differentiable step reproduces propagate_walkers + Cholesky re-orthogonalisation of functions.py
  - walkers = trial: dE/dlam = <V>_HF
  - no population control: forward-mode derivative = central finite difference with the same random numbers
  - forward mode = reverse mode (with Martin's rebalancing rule, with and without checkpointing)
  - two-body perturbation L + lam dL: the rebuilt pieces equal the setup of functions.py; walkers = trial give the
    finite difference of the HF energy of Hamiltonians built by functions.py at lam = +-h; forward = finite
    difference along a trajectory; forward = reverse
"""
import os
import sys
import numpy as np
import pytest
import scipy.linalg

sys.path.insert(0, os.path.dirname(__file__))
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import functions as new  # noqa: E402
import autodiff_forces as af  # noqa: E402
from test_compact_hamiltonian import _DATA_ROOT, COMPACT, make_config, perturbed  # noqa: E402

GAMMA = os.path.join(_DATA_ROOT, "test", "diamond_gamma_prim")
SYSTEMS = {"gamma": (GAMMA, dict(nk=1, nb=16, ne=4, ng=136)),      # one k-point: compact layout = dense layout
           "k222": (COMPACT, dict())}

NW = 6


@pytest.fixture(scope="module", params=list(SYSTEMS))
def system(request):
    path, sizes_kw = SYSTEMS[request.param]
    if not os.path.isdir(path):
        pytest.skip(f"test data {path} not found (set AFQMC_DATA)")
    config = make_config(NW, precision="Double", backend=new.JaxBackend(5), **sizes_kw)
    q = new.obtain_Q_list(config, os.path.join(path, "Q_list.npy"))
    h1 = new.obtain_H1(config, os.path.join(path, "H1_svd.npy"))
    h2c, _ = new.obtain_H2_host(config, os.path.join(path, "H2_zip.npy"))
    sizes = new.obtain_Q_sizes(config, os.path.join(path, "Q_sizes.npy"))
    H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    model = af.Model(config, H, trial, V=np.asarray(h1))            # V = H1: dE/dlam = <H1>
    walkers = perturbed(config, walkers)
    return config, H, trial, walkers, model


@pytest.fixture(scope="module", params=list(SYSTEMS))
def system2(request):
    """Two-body perturbation: dH1 and dL random (fixed seed), scaled to 10 % of H1 and L."""
    path, sizes_kw = SYSTEMS[request.param]
    if not os.path.isdir(path):
        pytest.skip(f"test data {path} not found (set AFQMC_DATA)")
    config = make_config(NW, precision="Double", backend=new.JaxBackend(5), **sizes_kw)
    files = dict(q=os.path.join(path, "Q_list.npy"), h1=os.path.join(path, "H1_svd.npy"),
                 h2=os.path.join(path, "H2_zip.npy"), sizes=os.path.join(path, "Q_sizes.npy"))
    h1 = np.asarray(new.obtain_H1(config, files["h1"]))
    h2c, _ = new.obtain_H2_host(config, files["h2"])
    rng = np.random.default_rng(17)
    dh1 = rng.standard_normal(h1.shape) * (h1 != 0)
    dh1 = 0.1 * np.max(np.abs(h1)) * (dh1 + dh1.T) / 2                  # real symmetric, same blocks as H1
    dL = 0.1 * np.max(np.abs(h2c)) * (rng.standard_normal(h2c.shape) + 1j * rng.standard_normal(h2c.shape))

    def hamiltonian(lam):
        q = new.obtain_Q_list(config, files["q"])
        sizes = new.obtain_Q_sizes(config, files["sizes"])
        H = new.build_hamiltonian(config, config.backend.array(h1 + lam * dh1), h2c + lam * dL, "compact", q, sizes)
        trial, _ = new.initialize_determinant(config)
        H.setup_energy_expressions(config, trial)
        return H

    H = hamiltonian(0.0)
    trial, walkers = new.initialize_determinant(config)
    model = af.Model(config, H, trial, V=dh1, L=h2c, dL=dL)
    walkers = perturbed(config, walkers)
    return config, H, trial, walkers, model, hamiltonian


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_expm_taylor_matches_scipy(system):
    config, H, trial, walkers, model = system
    a = -0.5 * config.timestep * np.asarray(model.h1_total)
    assert rel(af.expm_taylor(jnp.asarray(a)), scipy.linalg.expm(a)) < 1e-13


def test_step_matches_propagate_walkers(system):
    config, H, trial, walkers, model = system
    x = np.random.default_rng(3).standard_normal((2 * config.num_g, NW))
    H.test_random_field = jnp.asarray(x)
    e0 = float(new.measure_energy(config, trial, walkers, H)[0].real)
    h0 = np.exp(config.timestep * H.H_zero)
    ref, _ = new.propagate_walkers(config, trial, walkers, H, h0, e0)
    H.test_random_field = None
    ref_dets = new.reortho_qr(config, ref.slater_det, "Cholesky")
    exp_half = af.expm_taylor(-0.5 * model.dtau * model.h1_total)
    dets, w = model.step(exp_half, e0, walkers.slater_det, walkers.weights, jnp.asarray(x))
    assert rel(dets, ref_dets) < 1e-11
    assert rel(w, ref.weights) < 1e-11


def test_trial_walkers_give_hf_derivative(system):
    config, H, trial, walkers, model = system
    dets = jnp.asarray(np.array([np.asarray(trial)] * NW), dtype=model.ctype)
    w = jnp.ones(NW, dtype=model.ctype)
    e, de = jax.value_and_grad(lambda l: model.mixed_energy(l, dets, w))(0.0)
    hf = new.measure_energy(config, trial, new.Walkers(dets, w), H)[0].real
    assert abs(e - hf) < 1e-10 * abs(hf)
    assert abs(de - af.hf_derivative(trial, model.V)) < 1e-10 * abs(de)


def test_forward_derivative_equals_finite_difference_without_rebalancing(system):
    config, H, trial, walkers, model = system
    key = jax.random.PRNGKey(7)
    args = (walkers.slater_det, walkers.weights, 0.0, key, 3, 4, "off")
    e, de = model.energy_and_derivative(0.0, *args)
    h = 1e-4
    ep = model.window(h, *args)[0]
    em = model.window(-h, *args)[0]
    fd = (ep - em) / (2 * h)
    assert rel(model.window(0.0, *args)[0], e) < 1e-14
    assert rel(de, fd) < 1e-6


@pytest.mark.parametrize("mode", ["martin", "zero"])
def test_forward_equals_reverse(system, mode):
    config, H, trial, walkers, model = system
    key = jax.random.PRNGKey(11)
    args = (walkers.slater_det, walkers.weights, 0.0, key, 3, 3, mode)
    e, de = model.energy_and_derivative(0.3, *args)
    m, g = model.mean_energy_gradient(0.3, *args)
    mc, gc = model.mean_energy_gradient(0.3, *args, checkpoint=True)
    assert abs(m - np.mean(e)) < 1e-12 * abs(m)
    assert abs(g - np.mean(de)) < 1e-9 * abs(g)
    assert abs(gc - g) < 1e-12 * abs(g)


def test_two_body_pieces_match_setup(system2):
    config, H, trial, walkers, model, hamiltonian = system2
    for lam in (0.0, 0.3):
        Hl = hamiltonian(lam)
        P = model.pieces(lam)
        assert rel(P["M"], Hl._M) < 1e-13
        assert rel(P["c"], Hl._c) < 1e-13
        assert rel(P["G"], Hl._G) < 1e-13
        assert rel(P["h1_total"], -np.asarray(Hl._h1) / config.timestep) < 1e-13
        assert rel(P["h0"], np.exp(config.timestep * Hl.H_zero)) < 1e-13


def test_two_body_trial_walkers_give_hf_derivative(system2):
    """For walkers = trial the energy is the HF energy, quadratic in lam: the central difference of the energies of
    Hamiltonians set up by functions.py at lam = +-h is exact up to rounding."""
    config, H, trial, walkers, model, hamiltonian = system2
    dets = jnp.asarray(np.array([np.asarray(trial)] * NW), dtype=model.ctype)
    w = jnp.ones(NW, dtype=model.ctype)
    e, de = jax.value_and_grad(lambda l: model.mixed_energy(l, dets, w))(0.0)
    hf = lambda lam: float(new.measure_energy(config, trial, new.Walkers(dets, w), hamiltonian(lam))[0].real)
    h = 0.1
    assert abs(e - hf(0.0)) < 1e-11 * abs(e)
    assert abs(de - (hf(h) - hf(-h)) / (2 * h)) < 1e-9 * abs(de)


def test_two_body_forward_equals_finite_difference(system2):
    config, H, trial, walkers, model, hamiltonian = system2
    key = jax.random.PRNGKey(7)
    args = (walkers.slater_det, walkers.weights, 0.0, key, 3, 4, "off")
    e, de = model.energy_and_derivative(0.0, *args)
    h = 1e-4
    fd = (model.window(h, *args)[0] - model.window(-h, *args)[0]) / (2 * h)
    assert rel(de, fd) < 1e-6


@pytest.mark.parametrize("mode", ["martin", "zero"])
def test_two_body_forward_equals_reverse(system2, mode):
    config, H, trial, walkers, model, hamiltonian = system2
    key = jax.random.PRNGKey(11)
    args = (walkers.slater_det, walkers.weights, 0.0, key, 3, 3, mode)
    e, de = model.energy_and_derivative(0.1, *args)
    m, g = model.mean_energy_gradient(0.1, *args)
    mc, gc = model.mean_energy_gradient(0.1, *args, checkpoint=True)
    assert abs(m - np.mean(e)) < 1e-12 * abs(m)
    assert abs(g - np.mean(de)) < 1e-9 * abs(g)
    assert abs(gc - g) < 1e-12 * abs(g)
