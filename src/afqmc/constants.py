from dataclasses import dataclass
import numpy.typing as npt


@dataclass
class Constants:
    L: npt.ArrayLike
    number_electron: int
    number_walker: int
    number_k: int
    tau: float

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
