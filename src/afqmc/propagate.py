from opt_einsum import contract


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
