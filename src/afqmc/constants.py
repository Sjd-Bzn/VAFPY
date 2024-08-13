from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from opt_einsum import contract_expression
from scipy.linalg import expm


@dataclass
class Constants:
    """Stores all elements that remain constant throughout an AFQMC calculation
    and provides access to helper functions."""

    H1: npt.ArrayLike
    "Single-particle Hamiltonian contains kinetic energy, ionic and Hartree potential."
    L: npt.ArrayLike
    "Two-particle interaction after factorization, i.e., represented as sum of one-body operators."
    number_electron: int
    "Number of electrons in the system."
    number_walker: int
    "Number of walkers during the AFQMC propagation."
    number_k: int
    "Number of **k** points in the Brillouin zone."
    tau: float
    "Time step for each step in the propagation."
    propagate_order: int = 6
    "Expand the exponential operator up to this order."

    @property
    def number_orbital(self):
        "Size of the basis in the orbital representation."
        return self.L.shape[0]

    @property
    def number_g(self):
        "Number of interactions by which the two-body operator is represented."
        return self.L.shape[-1]

    @property
    def shape_field(self):
        "Matrix shape of a random field."
        return self.number_g, self.number_walker

    @property
    def shape_slater_det(self):
        "Tensor shape of all slater determinants."
        return self.number_walker, self.number_orbital, self.number_electron

    @property
    def sqrt_tau(self):
        "The square root of the time step."
        return np.sqrt(self.tau)

    @property
    def L_trial(self):
        "L projected on a trial determinant."
        return self.L[: self.number_electron]

    @property
    def exp_H1_half(self):
        "Precomputed value of exp(-H1 tau / 2)."
        return self._exp_H1_half

    def __post_init__(self):
        self._get_potential = contract_expression(
            "ijg,gw->ijw", self.L, self.shape_field, constants=[0]
        )
        self._get_force_bias = contract_expression(
            "wij,jig->gw", self.shape_slater_det, self.L_trial, constants=[1]
        )
        self._exp_H1_half = expm(-0.5 * self.tau * self.H1)
