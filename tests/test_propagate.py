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


# def test_propagate_s2():
#     num_g = 14
#     num_orbital = 8
#     num_electron = 2
#     num_walker = 6
#     num_k = 1
#     tau = 1e-4
#     L = np.random.random((num_orbital, num_orbital, num_g))
