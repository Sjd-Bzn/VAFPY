"""Exactness of the derivative of the AFQMC energy by automatic differentiation (autodiff_forces.py), C 2x2x2
primitive test data, double precision, JAX on the CPU.

  - the differentiable step reproduces propagate_walkers + Cholesky re-orthogonalisation of functions.py
  - walkers = trial: dE/dlam = <V>_HF
  - no population control: forward-mode derivative = central finite difference with the same random numbers
  - forward mode = reverse mode (with Martin's rebalancing rule, with and without checkpointing)
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
from test_compact_hamiltonian import COMPACT, make_config, perturbed  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(COMPACT), reason="k-point test data not found (set AFQMC_DATA)")

NW = 6


@pytest.fixture(scope="module")
def system():
    config = make_config(NW, precision="Double", backend=new.JaxBackend(5))
    q = new.obtain_Q_list(config, os.path.join(COMPACT, "Q_list.npy"))
    h1 = new.obtain_H1(config, os.path.join(COMPACT, "H1_svd.npy"))
    h2c, layout = new.obtain_H2_host(config, os.path.join(COMPACT, "H2_zip.npy"))
    sizes = new.obtain_Q_sizes(config, os.path.join(COMPACT, "Q_sizes.npy"))
    H = new.build_hamiltonian(config, h1, h2c, layout, q, sizes)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    model = af.Model(config, H, trial, V=np.asarray(h1))            # V = H1: dE/dlam = <H1>
    walkers = perturbed(config, walkers)
    return config, H, trial, walkers, model


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
