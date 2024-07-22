from dataclasses import dataclass
from opt_einsum import contract_expression
import numpy.typing as npt


@dataclass
class Constants:
    L: npt.ArrayLike
    number_walker: int


class Propagator:
    def __init__(self, constants):
        num_g = constants.L.shape[-1]
        self._make_auxiliary_field = contract_expression(
            "ijG,GN->ijN", constants.L, (num_g, constants.number_walker), constants=[0]
        )

    def new_auxiliary_field(self, force_bias):
        if force_bias is None:
            x = self.random_normal()
        else:
            x = self.random_normal() - force_bias
        return self._make_auxiliary_field(x)

    def random_normal(self):
        pass
