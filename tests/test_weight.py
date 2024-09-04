from cmath import phase

import numpy as np
import numpy.testing as npt

from afqmc import weight


def test_phaseless(make_constants):
    constants = make_constants()
    old_slater_det = np.random.random(constants.shape_slater_det) + 0.1j
    new_slater_det = np.random.random(constants.shape_slater_det) + 0.1j
    expected = []
    importance_sampling = 0.9 + 0.1j
    for old, new in zip(old_slater_det, new_slater_det):
        old_overlap = np.linalg.det(old[: constants.number_electron]) ** 2
        new_overlap = np.linalg.det(new[: constants.number_electron]) ** 2
        ratio = new_overlap / old_overlap
        phase_factor = max(0, np.cos(phase(ratio)))
        expected.append(np.abs(ratio * importance_sampling) * phase_factor)
    actual = weight.phaseless(
        constants, old_slater_det, new_slater_det, importance_sampling
    )
    npt.assert_allclose(actual, expected)
