import numpy as np
import numpy.testing as npt
from scipy.sparse import block_diag

from afqmc import energy

SPIN_DEGEN = 2


def test_HF_exchange(make_constants):
    constants = make_constants(number_k=8)
    L_occ = project_L_trial(constants)
    expected = 0.5 * SPIN_DEGEN * np.einsum("ijg,ijg->", L_occ, L_occ.conj())
    hf_det = constants.trial_det
    actual = energy.exchange(constants, hf_det[np.newaxis])
    npt.assert_allclose(actual, expected)


def test_HF_hartree(make_constants):
    constants = make_constants(number_k=6)
    L_occ = project_L_trial(constants)
    expected = SPIN_DEGEN * np.einsum("iig,jjg->", L_occ, L_occ.conj())
    hf_det = constants.trial_det
    actual = energy.hartree(constants, hf_det[np.newaxis])
    npt.assert_allclose(actual, expected)


def test_HF_one_particle(make_constants):
    constants = make_constants(number_k=4)
    H1 = block_diag(constants.H1)
    hf_det = constants.trial_det
    expected = SPIN_DEGEN * np.einsum("ni,nm,mi->", hf_det, H1.toarray(), hf_det)
    actual = energy.one_particle(constants, hf_det[np.newaxis])
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
    hf_det = constants.trial_det
    H1 = block_diag(constants.H1).toarray()
    one_particle = SPIN_DEGEN * np.einsum("ni,nm,wmi,w->", hf_det, H1, thetas, weight)
    norm = 1 / sum(weight)
    actual = energy.sample(constants, slater_det, weight)
    npt.assert_allclose(actual, norm * (exchange + hartree + one_particle))


def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial, walker)))


def overlap(left_slater_det, right_slater_det):
    return np.dot(left_slater_det.transpose(), right_slater_det)
