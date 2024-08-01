from opt_einsum import contract

from afqmc import determinant, field, weight


def time_step(constants, old_slater_det, old_weight):
    biorthogonal_det = determinant.biorthogonalize(constants, old_slater_det)
    force_bias = field.force_bias(constants, biorthogonal_det)
    auxiliary_field = field.auxiliary(constants, force_bias)
    potential = field.potential(constants, auxiliary_field)
    new_slater_det = s2(constants, potential, old_slater_det)
    I = field.importance_sampling(auxiliary_field, force_bias)
    new_weight = old_weight * weight.phaseless(old_slater_det, new_slater_det, I)
    return new_slater_det, new_weight


def taylor(constants, potential, slater_det):
    result = slater_det.copy()
    addend = slater_det.copy()
    for j in range(constants.propagate_order):
        addend = contract("ijw,wjk->wik", potential, addend) / (j + 1)
        result += addend
    return result


def s2(constants, potential, slater_det):
    slater_det = constants.exp_H1_half @ slater_det
    slater_det = taylor(constants, potential, slater_det)
    return constants.exp_H1_half @ slater_det
