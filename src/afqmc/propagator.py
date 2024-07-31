from dataclasses import dataclass
from opt_einsum import contract_expression
import numpy.typing as npt


@dataclass
class Constants:
    L: npt.ArrayLike
    number_electron: int
    number_walker: int

    @property
    def number_orbital(self):
        return self.L.shape[0]

    @property
    def number_g(self):
        return self.L.shape[-1]

    @property
    def shape_field(self):
        return self.number_g, self.number_walker

    @property
    def shape_slater_det(self):
        return self.number_walker, self.number_orbital, self.number_electron

    @property
    def L_trial(self):
        "L projected on a trial determinant"
        return self.L[: self.number_electron]


class Propagator:
    def __init__(self, constants):
        self._get_auxiliary_field = contract_expression(
            "ijg,gw->ijw", constants.L, constants.shape_field, constants=[0]
        )
        self._get_force_bias = contract_expression(
            "wij,jig->wg", constants.shape_slater_det, constants.L_trial, constants=[1]
        )

    def new_auxiliary_field(self, force_bias):
        if force_bias is None:
            x = self.random_normal()
        else:
            x = self.random_normal() - force_bias
        return self._get_auxiliary_field(x)

    def force_bias(self, slater_det):
        return self._get_force_bias(slater_det)

    def random_normal(self):
        pass
