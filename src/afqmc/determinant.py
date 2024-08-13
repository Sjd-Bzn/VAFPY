import numpy as np
from opt_einsum import contract


def project_trial(constants, slater_det):
    """Computes the overlap between the trial wavefunction and a Slater determinant.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    slater_det
        Current state (Slater determinant) of all walkers.

    Returns
    -------
    Projection of the Slater determinant on a trial state.

    Note
    ----
    This assumes the trial state is one on the diagonal and zero otherwise.
    """
    mask = np.zeros(constants.number_orbital * constants.number_k, dtype=np.bool_)
    for k in range(constants.number_k):
        first = constants.number_orbital * k
        last = first + constants.number_electron
        mask[first:last] = True
    return slater_det[:, mask]


def biorthogonolize(constants, slater_det):
    """Biortogonalize the orbital is the Slater determinant.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    slater_det
        Current state (Slater determinant) of all walkers.


    Returns
    -------
    Biorthogonal Slater determinant.
    """
    overlap = project_trial(constants, slater_det)
    inv_overlap = np.linalg.inv(overlap)
    return contract("wij,wjk->wik", slater_det, inv_overlap)


def overlap_trial(constants, slater_det):
    """Compute the overlap of the Slater determinants with the trial state.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    slater_det
        Current state (Slater determinant) of all walkers.

    Returns
    -------
    The overlap of the Slater determinant with the trial state.

    Note
    ----
    This assumes the trial state is one on the diagonal and zero otherwise.
    """
    overlap = project_trial(constants, slater_det)
    return np.linalg.det(overlap)
