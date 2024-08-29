import numpy as np
import numpy.testing as npt
from afqmc import energy


def test_HF_exchange(make_constants):
    constants = make_constants(number_k=8)
    number_empty = constants.number_orbital - constants.number_electron
    mask_single_kpoint = constants.number_electron * [True] + number_empty * [False]
    mask = np.array(constants.number_k * mask_single_kpoint)
    L_occ = constants.L[mask][:, mask]
    expected = np.einsum("ijg,ijg->", L_occ, L_occ.conj())
    hf_det = constants.trial_det
    actual = energy.exchange(constants, hf_det[np.newaxis])
    npt.assert_allclose(actual, expected)
