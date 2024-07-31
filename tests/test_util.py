import numpy as np
import numpy.testing as npt
from afqmc.constants import Constants
from afqmc.util import biorthogonolize


def test_biorthogonalize():
    num_g = 15
    num_orbital = 9
    num_electron = 2
    num_walker = 7
    num_k = 1
    tau = 1e-4
    L = np.random.random((num_orbital, num_orbital, num_g))
    trial = np.eye(num_orbital, num_electron)
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    slater_det = np.random.random(constants.shape_slater_det)
    expected = []
    for walker in slater_det:
        expected.append(theta(trial, walker))
    expected = np.array(expected)
    actual = biorthogonolize(constants, slater_det)
    npt.assert_allclose(expected, actual)


def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial, walker)))


def overlap(left_slater_det, right_slater_det):
    return np.dot(left_slater_det.transpose(), right_slater_det)
