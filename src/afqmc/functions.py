from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from opt_einsum import contract_expression


def main():
    config = Configuration(
        num_walkers=10,
        num_kpoint=1,
        num_orbital=8,
        num_electron=4,
        singularity=0,
        comm=MPI.COMM_WORLD,
        precision="foo",
    )
    hamiltonian = Hamiltonian(
        one_body=reshape_H1(config, np.load("H1.npy")),
        two_body=np.moveaxis(np.load("H2.npy"), 0, -1),
    )
    trial_det, walkers = initialize_determinant(config)
    hamiltonian.setup_energy_expressions(config, trial_det)

    E = measure_energy(config, trial_det, walkers, hamiltonian)
    expected = 631.88947 - 2.8974954e-09j
    print(E, np.isclose(E, expected))
    np.random.seed(1887431)
    walkers.slater_det += 0.05 * np.random.rand(*walkers.slater_det.shape)
    E = measure_energy(config, trial_det, walkers, hamiltonian)
    expected = 631.8965 - 0.009482565j
    print(E, np.isclose(E, expected))


@dataclass
class Configuration:
    num_walkers: int  # number of walkers per core
    num_kpoint: int  # number of k-points to sample the Brillouin zone
    num_orbital: int  # size of the basis
    num_electron: int  # number of occupied states
    singularity: float  # singularity used for the G=0 component (fsg)
    comm: MPI.Comm  # communicator over which the walkers are distributed
    precision: str  # must be either single or double

    @property
    def float_type(self):
        if self.precision == "single":
            return np.single
        elif self.precision == "double":
            return np.double
        else:
            raise NotImplementedError(f"Specified precision {self.precision} not implemented.")

    @property
    def complex_type(self):
        if self.precision == "single":
            return np.csingle
        elif self.precision == "double":
            return np.cdouble
        else:
            raise NotImplementedError(f"Specified precision {self.precision} not implemented.")


@dataclass
class Walkers:
    slater_det: np.typing.ArrayLike
    weights: np.typing.ArrayLike


@dataclass
class Hamiltonian:
    one_body: np.typing.ArrayLike
    # one-body part of Hamiltonian, expected shape (num_orbital, num_orbital, num_kpoint)
    two_body: np.typing.ArrayLike
    # two-body part of Hamiltonian after factorization,
    # expected shape (num_orbital, num_orbital * num_kpoint, num_g * num_kpoint)

    def setup_energy_expressions(self, config, trial_det):
        shape_theta = (
            config.num_walkers,
            config.num_orbital * config.num_kpoint,
            config.num_electron * config.num_kpoint,
        )
        b = np.dot(trial_det.T, self.one_body).astype(np.complex64)
        alp = np.einsum("pi, prg -> irg", trial_det, self.two_body)
        alp_t = np.einsum("pi, rpg -> irg", trial_det, self.two_body.conj())
        alp_s = alp.astype(np.complex64)
        alp_s_t = alp_t.astype(np.complex64)
        self._one_body_expression = contract_expression(
            "ip, wpi -> w", b, shape_theta, constants=[0], optimize="greedy"
        )
        args = (shape_theta, alp_s, shape_theta, alp_s_t)
        kwargs = {"constants": [1, 3], "optimize": "greedy"}
        self._hartree_expression = contract_expression(
            "wri, irg, wpj, jpg -> w", *args, **kwargs
        )
        self._exchange_expression = contract_expression(
            "wri, jrg, wpj, ipg -> w", *args, **kwargs
        )
        self._singularity_correction = (
            config.num_electron * config.num_kpoint * config.singularity
        )

    def compute_one_body(self, theta):
        return 2 * self._one_body_expression(theta)

    def compute_hartree(self, theta):
        return 2 * self._hartree_expression(theta, theta)

    def compute_exchange(self, theta):
        return -self._exchange_expression(theta, theta) - self._singularity_correction


def initialize_determinant(config):
    trial_det = np.eye(config.num_orbital, config.num_electron)
    walkers = Walkers(
        slater_det=np.array(config.num_walkers * [trial_det], dtype=np.complex64),
        weights=np.ones(config.num_walkers, dtype=np.complex64),
    )
    return trial_det, walkers


def reshape_H1(config, input_h1):
    """
    Reshape H1 (H1.npy shape is num_orb, num_orb, num_k
    we implement the k_points to the H1)
    """
    n = config.num_orbital * config.num_kpoint
    output_h1 = np.zeros((n, n), dtype=np.complex128)
    for i in range(config.num_kpoint):
        output_h1[
            i * config.num_orbital : (i + 1) * config.num_orbital,
            i * config.num_orbital : (i + 1) * config.num_orbital,
        ] = input_h1[:, :, i]
    return output_h1


def measure_energy(config, trial, walkers, hamiltonian):
    # compute energies locally
    theta = biorthogonalize(trial, walkers.slater_det)
    energy_one_body = hamiltonian.compute_one_body(theta)
    energy_hartree = hamiltonian.compute_hartree(theta)
    energy_exchange = hamiltonian.compute_exchange(theta)
    energy = (energy_one_body + energy_hartree + energy_exchange) / config.num_kpoint
    weighted_energy = energy @ walkers.weights

    # average over all ranks
    sum_weights = np.sum(walkers.weights)
    weighted_energy_global = config.comm.allreduce(weighted_energy)
    sum_weights_global = config.comm.allreduce(sum_weights)
    return weighted_energy_global / sum_weights_global


def biorthogonalize(trial, walkers):
    inverse_overlap = np.linalg.inv(trial.T @ walkers)
    return np.einsum("wpi, wij -> wpj", walkers, inverse_overlap)


if __name__ == "__main__":
    main()
