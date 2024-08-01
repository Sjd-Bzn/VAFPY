import numpy as np
from opt_einsum import contract


def potential(constants, auxiliary_field):
    return 1j * constants.sqrt_tau * constants._get_potential(auxiliary_field)

def auxiliary(constants, force_bias):
    if force_bias is None:
        return random(constants)
    else:
        return random(constants) - force_bias

def random(constants):
    return np.random.normal(size=constants.shape_field)

def force_bias(constants, slater_det):
    return -2j * constants.sqrt_tau * constants._get_force_bias(slater_det)

def importance_sampling(auxiliary_field, force_bias):
    argument = contract("gw->w", (auxiliary_field + 0.5 * force_bias) * force_bias)
    return np.exp(argument)
