import numpy as np

from afqmc import determinant


def phaseless(constants, old_slater_det, new_slater_det, importance_sampling):
    """Update factor for the weights in the phaseless approximation.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    old_slater_det
        Current state (Slater determinant) of all walkers.
    new_slater_det
        State after one time step, i.e., updated Slater determinants
    imporance_sampling
        Importance sampling factor to correct the update of the weights.

    Returns
    -------
    A factor one can use to scale the weights for the next step.
    """
    new_overlap = determinant.overlap_trial(constants, new_slater_det)
    old_overlap = determinant.overlap_trial(constants, old_slater_det)
    return np.maximum(0, (new_overlap / old_overlap).real) * np.abs(importance_sampling)
