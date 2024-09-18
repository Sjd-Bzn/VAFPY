import dataclasses
import itertools
from unittest.mock import patch

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
    H1 = -0.5 * constants.H1_full[0] * constants.tau
    U = expm(H1) @ expm(potential) @ expm(H1)
    expected = np.einsum("wij,wjk->wik", U, slater_det)
    npt.assert_allclose(actual, expected)


# def test_HF_energy(make_constants):
#     # in the limit of small timesteps exp(-Ht) = I - Ht so that <H> = (1 - <exp(-Ht)>) / t
#     number_walker = 100_000
#     tau = 1e-8
#     constants = make_constants(tau=tau, propagate_order=10, number_walker=number_walker, use_force_bias=False)
#     constants = dataclasses.replace(constants, H1 = np.zeros_like(constants.H1))
#     # constants = dataclasses.replace(constants, L=np.zeros_like(constants.L))
#     mat = np.random.random((constants.number_g, constants.number_g))
#     U, *_ = np.linalg.svd(mat)
#     example(constants, "svd", U)
#     example(constants, "diag", np.eye(constants.number_g))
#     slater_det = np.array(number_walker * [constants.trial_det], dtype=np.complex128)
#     weight = np.ones(number_walker)
#     expected = energy.sample(constants, slater_det, weight).real
#     slater_det, weight = propagate.time_step(constants, slater_det, weight)
#     overlap = determinant.overlap_trial(constants, slater_det)
#     actual = np.average((1 - overlap.real) / tau)
#     print(constants.number_orbital, constants.number_electron)
#     print(actual)
#     print(expected)
#     npt.assert_allclose(actual, expected)

# def test_HF_energy2(make_constants):

#     # in the limit of small timesteps exp(-Ht) = I - Ht so that <H> = (1 - <exp(-Ht)>) / t
#     number_walker = number_g = 100
#     tau = 1e-8
#     constants = make_constants(tau=tau, propagate_order=10, number_walker=number_walker, number_g=number_g, use_force_bias=False)
#     constants = dataclasses.replace(constants, H1 = np.zeros_like(constants.H1))
#     # constants = dataclasses.replace(constants, L=np.zeros_like(constants.L))
#     example(constants, "diag", np.eye(number_g))
#     slater_det = np.array(number_walker * [constants.trial_det], dtype=np.complex128)
#     weight = np.ones(number_walker)
#     expected = energy.sample(constants, slater_det, weight).real
#     with patch("afqmc.field.random", return_value=np.eye(number_g)):
#         slater_det, weight = propagate.time_step(constants, slater_det, weight)
#     overlap = determinant.overlap_trial(constants, slater_det)
#     actual = np.average((1 - overlap.real) / tau)
#     print(actual)
#     print(expected)
#     npt.assert_allclose(actual, expected)


# def example(constants, label, fields):
#     sum_ = 0
#     H1 = 0 #-constants.H1_full[0] * constants.tau
#     H2w = 1j * np.einsum("nmg,wg->wnm", constants.L, fields) * constants.sqrt_tau
#     for H2 in H2w:
#         slater_det = expm(H1 + H2) @ constants.trial_det
#         overlap = determinant.overlap_trial(constants, slater_det[np.newaxis])
#         sum_ += ((1 - overlap.real) / constants.tau)[0]
#     print(f"{label} {sum_ / len(fields)}")


# def test_explicit(make_constants):
#     number_walker = 10000
#     tau=1e-8
#     constants = make_constants(tau=tau, number_walker= number_walker, number_electron=3)
#     print(constants.number_electron)
#     Ex = 0
#     Eh = 0
#     Esic = 0
#     slater_det = np.array(number_walker * [constants.trial_det], dtype=np.complex128)
#     constants = dataclasses.replace(constants, H1 = np.zeros_like(constants.H1))
#     weight = np.ones(number_walker)
#     expected = energy.sample(constants, slater_det, weight).real
#     print(f"{expected=}")
#     for p,q,r,s in itertools.product(range(constants.number_electron), repeat=4):
#         if p == s and r == q:
#             Ex += np.einsum("g,g->", constants.L[p,r], constants.L[s,q].conj())
#         if p == r and s == q:
#             Eh += 2 * np.einsum("g,g->", constants.L[p,r], constants.L[s,q].conj())
#     for p,s in itertools.product(range(constants.number_electron), repeat=2):
#         if p == s:
#             Esic += 2 * np.einsum("ng,ng->", constants.L[p], constants.L[s].conj())
#     print(f"{Ex=} {Eh=} {Esic=}")
#     slater_det, weight = propagate.time_step(constants, slater_det, weight)
#     overlap = determinant.overlap_trial(constants, slater_det)
#     actual = np.average((1 - overlap.real) / tau)
#     print(f"{actual=}")
#     ref = np.einsum("nmg,")
#     x = np.eye(constants.number_g)
#     # x = np.random.normal(size=(constants.number_g, number_walker))
#     H = np.einsum("nmg,gw->wnm", constants.L, x)
#     H2psi = np.einsum("wnm,wml,li->wni", H, H, constants.trial_det)
#     print(f"{H2psi.shape=}")
#     overlap = determinant.overlap_trial(constants, H2psi)
#     print(f"{np.sum(overlap)=}")
#     print((expected - actual) / Ex)
#     assert False
def test_only_H1(make_constants):
    tau = 1e-8
    number_walker = 1
    constants = make_constants(tau=tau, number_walker=number_walker, L_zero=True)
    slater_det = np.array([constants.trial_det], dtype=np.complex128)
    weight = np.ones(number_walker)
    expected = energy.sample(constants, slater_det, weight).real
    slater_det, weight = propagate.time_step(constants, slater_det, weight)
    overlap = determinant.overlap_trial(constants, slater_det)
    actual = np.average((1 - overlap.real) / tau)
    npt.assert_allclose(actual, expected)


def test_only_L_no_force_bias(make_constants):
    tau = 1e-8
    number_g = 25
    number_walker = 100000
    constants = make_constants(
        tau=tau,
        number_g=number_g,
        number_walker=number_walker,
        H1_zero=True,
        use_force_bias=False,
    )
    slater_det = np.array(number_walker * [constants.trial_det], dtype=np.complex128)
    weight = np.ones(1)
    expected = energy.sample(constants, constants.trial_det[np.newaxis], weight).real
    # with patch("afqmc.field.random", return_value=np.eye(constants.number_g)):
    slater_det, weight = propagate.time_step(constants, slater_det, weight)
    print(slater_det.shape)
    overlap = determinant.overlap_trial(constants, slater_det)
    print(overlap.shape, sum(weight))
    actual = np.average((1 - overlap.real) / tau)
    Lf = constants.L_full
    Lf_occ = Lf[:constants.number_electron][:,:constants.number_electron]
    Eh = 2 * np.einsum("iig,jjg->", Lf_occ, Lf_occ)
    Ex = np.einsum("iag,aig->", Lf[:constants.number_electron][:,constants.number_electron:],
    Lf[constants.number_electron:][:,:constants.number_electron])
    Esic = np.einsum("iag,aig->", Lf[:constants.number_electron],
    Lf[:,:constants.number_electron])
    expected2 = Esic - Ex + Eh
    print(f"{Ex=} {Eh=} {Esic=} {expected2=}")
    print(f"{constants.number_electron=}")
    print(f"{actual=} {expected=} {expected2=}")
    np.save("H1.npy", constants.H1)
    np.save("L.npy", constants.L)
    np.save("H1_full.npy", constants.H1_full)
    np.save("L_full.npy", constants.L_full)
    npt.assert_allclose(actual, expected)
    assert False

### TODO: Amir used different L for energy and time propagation
