import numpy as np
from afqmc import determinant


def phaseless(constants, old_slater_det, new_slater_det, importance_sampling):
    new_overlap = determinant.overlap_trial(constants, new_slater_det)
    old_overlap = determinant.overlap_trial(constants, old_slater_det)
    return np.maximum(0, (new_overlap / old_overlap).real) * np.abs(importance_sampling)
