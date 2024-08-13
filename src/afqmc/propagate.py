from opt_einsum import contract

from afqmc import determinant, field, weight


def time_step(constants, old_slater_det, old_weight):
    """Propagate all Slater determinants and weights for one time step.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    old_slater_det
        Current state (Slater determinant) of all walkers.
    old_weight
        Weights of the current walkers.

    Returns
    -------
    new_slater_det
        State after one time step, i.e., updated Slater determinants
    new_weight
        Updated weights after the time step.
    """
    biorthogonal_det = determinant.biorthogonalize(constants, old_slater_det)
    force_bias = field.force_bias(constants, biorthogonal_det)
    auxiliary_field = field.auxiliary(constants, force_bias)
    potential = field.potential(constants, auxiliary_field)
    new_slater_det = s2(constants, potential, old_slater_det)
    I = field.importance_sampling(auxiliary_field, force_bias)
    new_weight = old_weight * weight.phaseless(old_slater_det, new_slater_det, I)
    return new_slater_det, new_weight


def s2(constants, potential, slater_det):
    """Split the propagation of H1 and potential to minimize the time-step error.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    potential
        Potential that is exponentiated and applied to the walkers.
    slater_det
        Current state (Slater determinant) of all walkers.

    Returns
    -------
    The exponentiated H1 + potential applied to all walkers.
    """
    slater_det = constants.exp_H1_half @ slater_det
    slater_det = taylor(constants, potential, slater_det)
    return constants.exp_H1_half @ slater_det


def taylor(constants, potential, slater_det):
    """Approximate the matrix exponential by a Taylor expansion.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    potential
        Potential that is exponentiated and applied to the walkers.
    slater_det
        Current state (Slater determinant) of all walkers.

    Returns
    -------
    The exponentiated potential applied to all walkers.
    """
    result = slater_det.copy()
    addend = slater_det.copy()
    for j in range(constants.propagate_order):
        addend = contract("ijw,wjk->wik", potential, addend) / (j + 1)
        result += addend
    return result
