from opt_einsum import contract


def taylor(constants, potential, slater_det):
    result = slater_det.copy()
    addend = slater_det.copy()
    for j in range(constants.propagate_order):
        addend = contract("ijw,wjk->wik", potential, addend) / (j + 1)
        result += addend
    return result
