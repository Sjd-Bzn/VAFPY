from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from opt_einsum import contract, contract_expression
from scipy.linalg import expm
from scipy.sparse import block_diag


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
    tau: float
    "Time step for each step in the propagation."
    propagate_order: int = 6
    "Expand the exponential operator up to this order."
    use_force_bias: bool = True
    "Should we use force bias in the calculations."

    # currently hard coded should depend on whether the system has spin or not
    spin_degeneracy = 2

    @property
    def number_k(self):
        "Number of **k** points in the Brillouin zone."
        return self.H1.shape[0]

    @property
    def number_orbital(self):
        "Size of the basis in the orbital representation."
        return self.H1.shape[1]

    @property
    def number_g(self):
        "Number of interactions by which the two-body operator is represented."
        return self.L.shape[-1] // self.number_k

    @property
    def shape_field(self):
        "Matrix shape of a random field."
        return self.L.shape[-1], self.number_walker

    @property
    def shape_slater_det(self):
        "Tensor shape of all slater determinants."
        return self.number_walker, *self._hf_det.shape

    @property
    def sqrt_tau(self):
        "The square root of the time step."
        return np.sqrt(self.tau)

    @property
    def L_trial(self):
        "L projected on a trial determinant."
        return self.L[self._occupied_mask]

    @property
    def H1_full(self):
        "In addition to H1, this contains also the mean-field subtraction and the self-interaction correction."
        return self._H1_full

    @property
    def exp_H1_half(self):
        "Precomputed value of exp(-H1 tau / 2)."
        return self._exp_H1_half

    @property
    def trial_det(self):
        "Trial determinant which is equivalent to the HF determinant."
        return self._hf_det

    def __post_init__(self):
        self._hf_det = hf_det = self._setup_hf_det()
        self._occupied_mask = self._setup_occupied_mask()
        self.get_potential = contract_expression(
            "ijg,gw->ijw", self.L, self.shape_field, constants=[0]
        )
        self.get_force_bias = contract_expression(
            "wij,jig->gw", self.shape_slater_det, self.L_trial, constants=[1]
        )
        self.get_exchange, self.get_hartree = self._setup_hartree_and_exchange(hf_det)
        self._H1_full = self._setup_H1_full()
        self.get_one_particle = self._setup_one_particle(hf_det)
        self._exp_H1_half = expm(-0.5 * self.tau * self._H1_full)

    def _setup_hf_det(self):
        hf_det_kpoint = np.eye(self.number_orbital, self.number_electron)
        return block_diag(self.number_k * [hf_det_kpoint]).toarray()

    def _setup_occupied_mask(self):
        number_empty = self.number_orbital - self.number_electron
        mask_single_kpoint = self.number_electron * [True] + number_empty * [False]
        return np.array(self.number_k * mask_single_kpoint)

    def _setup_hartree_and_exchange(self, hf_det):
        alpha = contract("ni,nmg->img", hf_det, self.L)
        beta = contract("ni,mng->img", hf_det, self.L.conj())
        exchange_expression = "wni,jng,wmj,img->w"
        hartree_expression = "wni,ing,wmj,jmg->w"
        return (
            contract_expression(
                expression,
                self.shape_slater_det,
                alpha,
                self.shape_slater_det,
                beta,
                constants=[1, 3],
                optimize="greedy",
            )
            for expression in (exchange_expression, hartree_expression)
        )

    def _setup_one_particle(self, hf_det):
        # H1_trial = contract("ni,nm->im", hf_det, block_diag(self.H1).toarray())
        H1_trial = contract("ni,nm->im", hf_det, block_diag(self._H1_full).toarray())
        return contract_expression(
            "im,wmi->w",
            H1_trial,
            self.shape_slater_det,
            constants=[0],
            optimize="greedy",
        )

    def _setup_H1_full(self):
        # TODO: implement k-point version
        if self.number_k == 1:
            SIC = contract("nmg,lmg->nl", self.L, self.L.conj())
        else:
            SIC = 0
        return self.H1 + SIC
