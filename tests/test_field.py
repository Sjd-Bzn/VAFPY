from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from afqmc import field
from afqmc.constants import Constants


def test_create_auxiliary_field_without_force_bias():
    num_g = 24
    num_orbital = 4
    num_electron = 1
    num_walker = 3
    num_k = 1
    tau = 3e-3
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    x = np.random.normal(size=constants.shape_field)
    with patch.object(field, "random", return_value=x) as mock:
        auxiliary_field = field.auxiliary(constants, force_bias=None)
        mock.assert_called_once_with(constants)
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
    x = np.random.normal(size=constants.shape_field)
    force_bias = np.random.random(constants.shape_field)
    with patch.object(field, "random", return_value=x) as mock:
        auxiliary_field = field.auxiliary(constants, force_bias)
        mock.assert_called_once_with(constants)
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
    force_bias = field.force_bias(constants, slater_det)
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
    force_bias = field.force_bias(constants, slater_det)
    auxiliary_field = field.auxiliary(constants, force_bias)
    potential = field.potential(constants, auxiliary_field)
    npt.assert_allclose(potential, 1j * np.sqrt(tau) * L @ auxiliary_field)


def test_importance_sampling():
    num_g = 17
    num_orbital = 4
    num_electron = 1
    num_walker = 5
    num_k = 1
    tau = 1e-4
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    slater_det = np.random.random(constants.shape_slater_det)
    x = np.random.normal(size=constants.shape_field)
    force_bias = field.force_bias(constants, slater_det)
    with patch.object(field, "random", return_value=x):
        auxiliary_field = field.auxiliary(constants, force_bias)
    importance_sampling = field.importance_sampling(auxiliary_field, force_bias)
    arg = np.einsum("gw->w", x * force_bias - 0.5 * force_bias * force_bias)
    npt.assert_allclose(importance_sampling, np.exp(arg))
