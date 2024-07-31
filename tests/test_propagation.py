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
    x = np.random.normal(size=(num_g, num_walker))
    with patch.object(propagator, "random_field", return_value=x) as mock:
        auxiliary_field = propagator.new_auxiliary_field(force_bias=None)
        mock.assert_called_once_with()
    npt.assert_allclose(auxiliary_field, 1j * np.sqrt(tau) * L @ x)


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
    x = np.random.normal(size=(num_g, num_walker))
    force_bias = np.random.random((num_g, num_walker))
    with patch.object(propagator, "random_field", return_value=x) as mock:
        auxiliary_field = propagator.new_auxiliary_field(force_bias)
        mock.assert_called_once_with()
    npt.assert_allclose(auxiliary_field, 1j * np.sqrt(tau) * L @ (x - force_bias))


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
    expected = -2j * np.sqrt(tau) * np.einsum("wij,jig->wg", slater_det, L_trial)
    npt.assert_allclose(force_bias, expected)


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
    aux_field = propagator.new_auxiliary_field(force_bias=None)
    slater_det = np.random.random(constants.shape_slater_det).astype(np.complex_)
    actual = propagator.propagate(aux_field, slater_det)
    aux_field = np.moveaxis(aux_field, -1, 0)
    expected = np.einsum("wij,wjk->wik", expm(aux_field), slater_det)
    npt.assert_allclose(actual, expected)
