import dataclasses

import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import expm

from afqmc import determinant, energy, field, propagate


def test_propagate_taylor(make_constants):
    constants = make_constants(tau=1e-4, propagate_order=10)
    slater_det = np.random.random(constants.shape_slater_det).astype(np.complex128)
    auxiliary_field = field.auxiliary(constants, force_bias=None)
    potential = field.potential(constants, auxiliary_field)
    actual = propagate.taylor(constants, potential, slater_det)
    potential = np.moveaxis(potential, -1, 0)
    expected = np.einsum("wij,wjk->wik", expm(potential), slater_det)
    npt.assert_allclose(actual, expected)


def test_propagate_s2(make_constants):
    constants = make_constants(tau=1e-4, propagate_order=10)
    slater_det = np.random.random(constants.shape_slater_det).astype(np.complex128)
    auxiliary_field = field.auxiliary(constants, force_bias=None)
    potential = field.potential(constants, auxiliary_field)
    actual = propagate.s2(constants, potential, slater_det)
    potential = np.moveaxis(potential, -1, 0)
    H1 = -0.5 * constants.H1[0] * constants.tau
    U = expm(H1) @ expm(potential) @ expm(H1)
    expected = np.einsum("wij,wjk->wik", U, slater_det)
    npt.assert_allclose(actual, expected)


def test_HF_energy(make_constants):
    # in the limit of small timesteps exp(-Ht) = I - Ht so that <H> = (1 - <exp(-Ht)>) / t
    number_walker = 1000
    tau = 1e-8
    constants = make_constants(tau=tau, propagate_order=10, number_walker=number_walker)
    # constants = dataclasses.replace(constants, H1 = np.zeros_like(constants.H1))
    constants = dataclasses.replace(constants, L=np.zeros_like(constants.L))
    slater_det = np.array(number_walker * [constants.trial_det], dtype=np.complex128)
    weight = np.ones(number_walker)
    expected = energy.sample(constants, slater_det, weight)
    slater_det, weight = propagate.time_step(constants, slater_det, weight)
    overlap = determinant.overlap_trial(constants, slater_det)
    actual = np.average((1 - overlap) / tau)
    npt.assert_allclose(actual, expected)
