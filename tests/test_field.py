from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from afqmc import field


def test_create_auxiliary_field_without_force_bias(make_constants):
    constants = make_constants()
    x = np.random.normal(size=constants.shape_field)
    with patch.object(field, "random", return_value=x) as mock:
        auxiliary_field = field.auxiliary(constants, force_bias=None)
        mock.assert_called_once_with(constants)
    npt.assert_allclose(auxiliary_field, x)


def test_create_auxiliary_field_with_force_bias(make_constants):
    constants = make_constants()
    x = np.random.normal(size=constants.shape_field)
    force_bias = np.random.random(constants.shape_field)
    with patch.object(field, "random", return_value=x) as mock:
        auxiliary_field = field.auxiliary(constants, force_bias)
        mock.assert_called_once_with(constants)
    npt.assert_allclose(auxiliary_field, x - force_bias)


def test_force_bias(make_constants):
    constants = make_constants()
    slater_det = np.random.random(constants.shape_slater_det)
    force_bias = field.force_bias(constants, slater_det)
    L_trial = constants.L[: constants.number_electron]
    expected = -2j * constants.sqrt_tau * np.einsum("wij,jig->gw", slater_det, L_trial)
    npt.assert_allclose(force_bias, expected)


def test_potential(make_constants):
    constants = make_constants()
    slater_det = np.random.random(constants.shape_slater_det)
    force_bias = field.force_bias(constants, slater_det)
    auxiliary_field = field.auxiliary(constants, force_bias)
    actual = field.potential(constants, auxiliary_field)
    expected = 1j * constants.sqrt_tau * constants.L @ auxiliary_field
    npt.assert_allclose(actual, expected)


def test_importance_sampling(make_constants):
    constants = make_constants()
    slater_det = np.random.random(constants.shape_slater_det)
    x = np.random.normal(size=constants.shape_field)
    force_bias = field.force_bias(constants, slater_det)
    with patch.object(field, "random", return_value=x):
        auxiliary_field = field.auxiliary(constants, force_bias)
    importance_sampling = field.importance_sampling(auxiliary_field, force_bias)
    arg = np.einsum("gw->w", x * force_bias - 0.5 * force_bias * force_bias)
    npt.assert_allclose(importance_sampling, np.exp(arg))
