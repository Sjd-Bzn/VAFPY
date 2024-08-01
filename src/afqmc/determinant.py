import numpy as np
from opt_einsum import contract


def project_trial(constants, slater_det):
    """
    It computes the overlap between the trial wavefunction and a Slater determinant.
    """
    mask = np.zeros(constants.number_orbital * constants.number_k, dtype=np.bool_)
    for k in range(constants.number_k):
        first = constants.number_orbital * k
        last = first + constants.number_electron
        mask[first:last] = True
    return slater_det[:, mask]


def biorthogonolize(constants, slater_det):
    overlap = project_trial(constants, slater_det)
    inv_overlap = np.linalg.inv(overlap)
    return contract("wij,wjk->wik", slater_det, inv_overlap)


def overlap_trial(constants, slater_det):
    overlap = project_trial(constants, slater_det)
    return np.linalg.det(overlap)
