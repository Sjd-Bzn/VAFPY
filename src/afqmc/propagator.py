from opt_einsum import contract_expression
import numpy as np


class Propagator:
    def __init__(self, constants):
        self._get_auxiliary_field = contract_expression(
            "ijg,gw->ijw", constants.L, constants.shape_field, constants=[0]
        )
        self._get_force_bias = contract_expression(
            "wij,jig->wg", constants.shape_slater_det, constants.L_trial, constants=[1]
        )
        self._sqrt_tau = np.sqrt(constants.tau)

    def new_auxiliary_field(self, force_bias):
        if force_bias is None:
            x = self.random_normal()
        else:
            x = self.random_normal() - force_bias
        return self._get_auxiliary_field(x)

    def force_bias(self, slater_det):
        return -2j * self._sqrt_tau * self._get_force_bias(slater_det)

    def random_normal(self):
        pass
