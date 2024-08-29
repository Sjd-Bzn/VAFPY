import numpy as np
import numpy.testing as npt
from afqmc import energy


def test_HF_exchange(make_constants):
    constants = make_constants(number_k=8)
    L_occ = project_L_trial(constants)
    expected = np.einsum("ijg,ijg->", L_occ, L_occ.conj())
    hf_det = constants.trial_det
    actual = energy.exchange(constants, hf_det[np.newaxis])
    npt.assert_allclose(actual, expected)


def test_HF_hartree(make_constants):
    constants = make_constants(number_k=6)
    L_occ = project_L_trial(constants)
    expected = 2 * np.einsum("iig,jjg->", L_occ, L_occ.conj())
    hf_det = constants.trial_det
    actual = energy.hartree(constants, hf_det[np.newaxis])
    npt.assert_allclose(actual, expected)


def project_L_trial(constants):
    number_empty = constants.number_orbital - constants.number_electron
    mask_single_kpoint = constants.number_electron * [True] + number_empty * [False]
    mask = np.array(constants.number_k * mask_single_kpoint)
    return constants.L[mask][:, mask]


def test_energy_walker(make_constants):
    constants = make_constants(number_k=2, number_orbital=5)
    slater_det = np.random.random(constants.shape_slater_det).astype(np.complex128)
    weight = np.random.random(constants.number_walker)
    thetas = []
    for walker in slater_det:
        thetas.append(theta(constants.trial_det, walker))
    thetas = np.array(thetas)
    Vij = np.einsum("nj,nmg,wmi->wijg", constants.trial_det, constants.L, thetas)
    Wij = np.einsum("ni,mng,wmj->wijg", constants.trial_det, constants.L.conj(), thetas)
    exchange = np.einsum("wijg,wijg,w", Vij, Wij, weight)
    hartree = 2 * np.einsum("wiig,wjjg,w", Vij, Wij, weight)
    actual = energy.sample(constants, slater_det, weight)
    npt.assert_allclose(actual, exchange + hartree)


def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial, walker)))


def overlap(left_slater_det, right_slater_det):
    return np.dot(left_slater_det.transpose(), right_slater_det)
