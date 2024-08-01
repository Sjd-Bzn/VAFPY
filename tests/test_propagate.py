import numpy as np
import numpy.testing as npt
from scipy.linalg import expm

from afqmc import field, propagate


def test_propagate_taylor(make_constants):
    constants = make_constants(tau=1e-4, propagate_order=10)
    slater_det = np.random.random(constants.shape_slater_det).astype(np.complex_)
    auxiliary_field = field.auxiliary(constants, force_bias=None)
    potential = field.potential(constants, auxiliary_field)
    actual = propagate.taylor(constants, potential, slater_det)
    potential = np.moveaxis(potential, -1, 0)
    expected = np.einsum("wij,wjk->wik", expm(potential), slater_det)
    npt.assert_allclose(actual, expected)


def test_propagate_s2(make_constants):
    constants = make_constants(tau=1e-4, propagate_order=10)
    slater_det = np.random.random(constants.shape_slater_det).astype(np.complex_)
    auxiliary_field = field.auxiliary(constants, force_bias=None)
    potential = field.potential(constants, auxiliary_field)
    actual = propagate.s2(constants, potential, slater_det)
    potential = np.moveaxis(potential, -1, 0)
    H1 = -0.5 * constants.H1[0] * constants.tau
    U = expm(H1) @ expm(potential) @ expm(H1)
    expected = np.einsum("wij,wjk->wik", U, slater_det)
    npt.assert_allclose(actual, expected)
