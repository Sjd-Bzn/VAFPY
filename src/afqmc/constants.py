from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from opt_einsum import contract_expression


@dataclass
class Constants:
    L: npt.ArrayLike
    number_electron: int
    number_walker: int
    number_k: int
    tau: float
    propagate_order: int = 6

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
    def sqrt_tau(self):
        return np.sqrt(self.tau)

    @property
    def L_trial(self):
        "L projected on a trial determinant"
        return self.L[: self.number_electron]

    def __post_init__(self):
        self._get_potential = contract_expression(
            "ijg,gw->ijw", self.L, self.shape_field, constants=[0]
        )
        self._get_force_bias = contract_expression(
            "wij,jig->gw", self.shape_slater_det, self.L_trial, constants=[1]
        )
