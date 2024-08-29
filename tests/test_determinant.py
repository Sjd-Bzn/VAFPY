import numpy as np
import numpy.testing as npt

from afqmc import determinant


def test_biorthogonalize(make_constants):
    constants = make_constants()
    slater_det = np.random.random(constants.shape_slater_det)
    expected = []
    for walker in slater_det:
        expected.append(theta(constants.trial_det, walker))
    expected = np.array(expected)
    actual = determinant.biorthogonolize(constants, slater_det)
    npt.assert_allclose(expected, actual, atol=1e-12)


def test_overlap_trial(make_constants):
    constants = make_constants()
    slater_det = np.random.random(constants.shape_slater_det)
    expected = np.linalg.det(slater_det[:, : constants.number_electron])
    npt.assert_allclose(determinant.overlap_trial(constants, slater_det), expected)


def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial, walker)))


def overlap(left_slater_det, right_slater_det):
    return np.dot(left_slater_det.transpose(), right_slater_det)
