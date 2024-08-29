import itertools

import numpy as np
import numpy.testing as npt


def test_size_properties(make_constants):
    number_k = 2
    constants = make_constants(number_k=number_k)
    assert constants.number_k == number_k
    assert constants.number_g == constants.L.shape[-1] // number_k
    assert constants.number_orbital == constants.H1.shape[1]


def test_shape_properties(make_constants):
    number_k = 3
    constants = make_constants(number_k=number_k)
    assert constants.shape_field == (constants.L.shape[-1], constants.number_walker)
    shape_slater_det = (
        constants.number_walker,
        number_k * constants.number_orbital,
        number_k * constants.number_electron,
    )
    assert constants.shape_slater_det == shape_slater_det


def test_trial_determinant(make_constants):
    number_k = 4
    constants = make_constants(number_k=number_k)
    hf_det = constants.trial_det
    loop_over_band_and_kpoint = itertools.product(
        range(constants.number_orbital),
        range(number_k),
        range(constants.number_electron),
        range(number_k),
    )
    for n, k, i, kp in loop_over_band_and_kpoint:
        nk = k * constants.number_orbital + n
        ikp = kp * constants.number_electron + i
        if k == kp and i == n:
            assert hf_det[nk, ikp] == 1
        else:
            assert hf_det[nk, ikp] == 0


def test_projection_on_trial_determinant(make_constants):
    constants = make_constants(number_k=3)
    expected = np.einsum("ni,nmg->img", constants.trial_det, constants.L)
    npt.assert_allclose(constants.L_trial, expected)
