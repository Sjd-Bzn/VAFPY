from afqmc import determinant


def sample(constants, slater_det):
    theta = determinant.biorthogonolize(constants, slater_det)
    return exchange(constants, theta)


def exchange(constants, slater_det):
    return constants.get_exchange(slater_det, slater_det)
