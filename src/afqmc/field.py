import numpy as np
from opt_einsum import contract


def potential(constants, auxiliary_field):
    """Computes the potential resulting from an auxiliary field.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    auxiliary_field
        Auxiliary field composed of random field and force bias.

    Returns
    -------
    Potential in matrix representation.
    """
    return 1j * constants.sqrt_tau * constants.get_potential(auxiliary_field)


def auxiliary(constants, force_bias):
    """Construct random field and combine with force bias (if present).

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    force_bias
        Force bias added to the random field. If set to None only the random
        field is constructed.

    Returns
    -------
    Auxiliary field composed of random field and force bias.
    """
    if force_bias is None:
        return random(constants)
    else:
        return random(constants) - force_bias


def random(constants):
    """Sample a random field from the normal distribution.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.

    Returns
    -------
    A normal-distributed random field of the appropriate size
    """
    return np.random.normal(size=constants.shape_field)


def force_bias(constants, slater_det):
    """Computes the force bias to minimize the imaginary part of the walkers.

    Parameters
    ----------
    constants
        Constants during the AFQMC calculation.
    slater_det
        Current state (Slater determinant) of all walkers.

    Returns
    -------
    Force bias to minimize rotation to the complex plane for the Slater determinants.
    """
    return -2j * constants.sqrt_tau * constants.get_force_bias(slater_det)


def importance_sampling(auxiliary_field, force_bias):
    """Computes the importance sampling factor for the weights.

    Parameters
    ----------
    auxiliary_field
        Auxiliary field composed of random field and force bias.
    force_bias
        Force bias to minimize rotation to the complex plane for the Slater determinants.

    Returns
    -------
    Importance sampling factor to correct the update of the weights.
    """
    argument = contract("gw->w", (auxiliary_field + 0.5 * force_bias) * force_bias)
    return np.exp(argument)
