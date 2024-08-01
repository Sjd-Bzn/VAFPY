from opt_einsum import contract, contract_expression
import numpy as np


class Propagator:
    ORDER_TAYLOR = 10

    def __init__(self, constants):
        self._shape_field = constants.shape_field
        self._get_potential = contract_expression(
            "ijg,gw->ijw", constants.L, constants.shape_field, constants=[0]
        )
        self._get_force_bias = contract_expression(
            "wij,jig->gw", constants.shape_slater_det, constants.L_trial, constants=[1]
        )
        self._sqrt_tau = np.sqrt(constants.tau)

    def propagate(self, potential, slater_det):
        result = slater_det.copy()
        addend = slater_det.copy()
        for j in range(self.ORDER_TAYLOR):
            addend = contract("ijw,wjk->wik", potential, addend) / (j + 1)
            result += addend
        return result

    def potential(self, auxiliary_field):
        return 1j * self._sqrt_tau * self._get_potential(auxiliary_field)

    def auxiliary_field(self, force_bias):
        if force_bias is None:
            return self.random_field()
        else:
            return self.random_field() - force_bias

    def random_field(self):
        return np.random.normal(size=self._shape_field)

    def force_bias(self, slater_det):
        return -2j * self._sqrt_tau * self._get_force_bias(slater_det)

    def importance_sampling(self, auxiliary_field, force_bias):
        argument = contract("gw->w", (auxiliary_field + 0.5 * force_bias) * force_bias)
        return np.exp(argument)
