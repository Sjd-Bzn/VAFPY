from afqmc import determinant


def sample(constants, slater_det, weight):
    theta = determinant.biorthogonalize(constants, slater_det)
    Ex = exchange(constants, theta)
    Eh = hartree(constants, theta)
    E1 = one_particle(constants, theta)
    print(f"{Ex[0]=} {Eh[0]=} {E1[0]=}")
    return (Ex + Eh + E1) @ weight / sum(weight)


def exchange(constants, slater_det):
    # TODO check sign of exchange
    energy = constants.get_exchange(slater_det, slater_det)
    return -0.5 * constants.spin_degeneracy * energy


def hartree(constants, slater_det):
    return constants.spin_degeneracy * constants.get_hartree(slater_det, slater_det)


def one_particle(constants, slater_det):
    return constants.spin_degeneracy * constants.get_one_particle(slater_det)
