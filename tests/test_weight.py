from cmath import phase

import numpy as np
import numpy.testing as npt

from afqmc.constants import Constants
from afqmc import weight


def test_phaseless():
    num_g = 22
    num_orbital = 8
    num_electron = 3
    num_walker = 10
    num_k = 1
    tau = 1e-4
    L = np.random.random((num_orbital, num_orbital, num_g))
    constants = Constants(L, num_electron, num_walker, num_k, tau)
    old_slater_det = np.random.random(constants.shape_slater_det) + 0.1j
    new_slater_det = np.random.random(constants.shape_slater_det) + 0.1j
    expected = []
    importance_sampling = 0.9 + 0.1j
    for old, new in zip(old_slater_det, new_slater_det):
        ratio = np.linalg.det(new[:num_electron]) / np.linalg.det(old[:num_electron])
        phase_factor = max(0, np.cos(phase(ratio)))
        expected.append(np.abs(ratio * importance_sampling) * phase_factor)
    actual = weight.phaseless(
        constants, old_slater_det, new_slater_det, importance_sampling
    )
    npt.assert_allclose(actual, expected)
