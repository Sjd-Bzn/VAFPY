from afqmc import determinant


def sample(constants, slater_det, weight):
    theta = determinant.biorthogonolize(constants, slater_det)
    energy = exchange(constants, theta) + hartree(constants, theta)
    return energy @ weight


def exchange(constants, slater_det):
    return constants.get_exchange(slater_det, slater_det)


def hartree(constants, slater_det):
    return 2 * constants.get_hartree(slater_det, slater_det)
