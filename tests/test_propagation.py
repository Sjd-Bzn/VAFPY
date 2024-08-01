from unittest.mock import patch
import numpy as np
import numpy.testing as npt
from scipy.linalg import expm
from afqmc.constants import Constants
from afqmc.propagator import Propagator


def test_create_auxiliary_field_without_force_bias():
    num_g = 24
    num_orbital = 4
    num_electron = 1
    num_walker = 3
    num_k = 1
    tau = 3e-3
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    propagator = Propagator(constants)
    x = np.random.normal(size=constants.shape_field)
    with patch.object(propagator, "random_field", return_value=x) as mock:
        auxiliary_field = propagator.auxiliary_field(force_bias=None)
        mock.assert_called_once_with()
    npt.assert_allclose(auxiliary_field, x)


def test_create_auxiliary_field_with_force_bias():
    num_g = 12
    num_orbital = 7
    num_electron = 2
    num_walker = 4
    num_k = 1
    tau = 1e-4
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    propagator = Propagator(constants)
    x = np.random.normal(size=constants.shape_field)
    force_bias = np.random.random(constants.shape_field)
    with patch.object(propagator, "random_field", return_value=x) as mock:
        auxiliary_field = propagator.auxiliary_field(force_bias)
        mock.assert_called_once_with()
    npt.assert_allclose(auxiliary_field, x - force_bias)


def test_force_bias():
    num_g = 18
    num_orbital = 5
    num_electron = 3
    num_walker = 5
    num_k = 1
    tau = 1e-3
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    assert constants.shape_slater_det == (num_walker, num_orbital, num_electron)
    slater_det = np.random.random(constants.shape_slater_det)
    propagator = Propagator(constants)
    force_bias = propagator.force_bias(slater_det)
    L_trial = L[:num_electron]
    expected = -2j * np.sqrt(tau) * np.einsum("wij,jig->gw", slater_det, L_trial)
    npt.assert_allclose(force_bias, expected)


def test_potential():
    num_g = 19
    num_orbital = 4
    num_electron = 2
    num_walker = 8
    num_k = 1
    tau = 2e-3
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    slater_det = np.random.random(constants.shape_slater_det)
    propagator = Propagator(constants)
    force_bias = propagator.force_bias(slater_det)
    auxiliary_field = propagator.auxiliary_field(force_bias)
    potential = propagator.potential(auxiliary_field)
    npt.assert_allclose(potential, 1j * np.sqrt(tau) * L @ auxiliary_field)


def test_propagate():
    num_g = 13
    num_orbital = 6
    num_electron = 3
    num_walker = 7
    num_k = 1
    tau = 3e-4
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    propagator = Propagator(constants)
    slater_det = np.random.random(constants.shape_slater_det).astype(np.complex_)
    auxiliary_field = propagator.auxiliary_field(force_bias=None)
    potential = propagator.potential(auxiliary_field)
    actual = propagator.propagate(potential, slater_det)
    potential = np.moveaxis(potential, -1, 0)
    expected = np.einsum("wij,wjk->wik", expm(potential), slater_det)
    npt.assert_allclose(actual, expected)


def test_importance_sampling():
    num_g = 17
    num_orbital = 4
    num_electron = 1
    num_walker = 5
    num_k = 1
    tau = 1e-4
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    propagator = Propagator(constants)
    slater_det = np.random.random(constants.shape_slater_det)
    x = np.random.normal(size=constants.shape_field)
    force_bias = propagator.force_bias(slater_det)
    with patch.object(propagator, "random_field", return_value=x):
        auxiliary_field = propagator.auxiliary_field(force_bias)
    importance_sampling = propagator.importance_sampling(auxiliary_field, force_bias)
    arg = np.einsum("gw->w", x * force_bias - 0.5 * force_bias * force_bias)
    npt.assert_allclose(importance_sampling, np.exp(arg))
