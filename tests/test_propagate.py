import numpy as np
import numpy.testing as npt
from scipy.linalg import expm

from afqmc import field, propagate
from afqmc.constants import Constants


def test_propagate_taylor():
    num_g = 13
    num_orbital = 6
    num_electron = 3
    num_walker = 7
    num_k = 1
    tau = 3e-4
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau, propagate_order=8)
    slater_det = np.random.random(constants.shape_slater_det).astype(np.complex_)
    auxiliary_field = field.auxiliary(constants, force_bias=None)
    potential = field.potential(constants, auxiliary_field)
    actual = propagate.taylor(constants, potential, slater_det)
    potential = np.moveaxis(potential, -1, 0)
    expected = np.einsum("wij,wjk->wik", expm(potential), slater_det)
    npt.assert_allclose(actual, expected)
