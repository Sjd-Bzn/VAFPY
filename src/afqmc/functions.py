from dataclasses import dataclass
import types

import numpy as np
import jax
import jax.numpy as jnp
import scipy
from mpi4py import MPI
from opt_einsum import contract, contract_expression

from scipy.linalg import expm

@dataclass
class HAMILTONIAN:
    one_body: np.complex128
    two_body: np.complex128
@dataclass
class HAMILTONIAN_MF:
    zero_body: np.complex128
    one_body: np.complex128
    two_body_e: np.complex128
    two_body_o: np.complex128

def read_datafile(filename):
    '''
    Read the data from the given file into a numpy array.
    '''
    return np.load(filename)

def reshape_H1(H1, num_k, num_orb):
    '''
    Reshape H1 (H1.npy shape is num_orb, num_orb, num_k
    we implement the k_points to the H1)
    '''
    h1=np.zeros([num_orb*num_k,num_orb*num_k],dtype=np.complex128)
    for i in range(0,num_k):
        h1[i*num_orb:(i+1)*num_orb,i*num_orb:(i+1)*num_orb] = H1[:,:,i]
    return h1

def A_af_MF_sub(trial_0,trial,h2,q_list,num_k,num_orb,num_electrons_up):
    '''
    It returns two body Hamiltonian after mean-field subtraction.
    '''
    avg_A_mat = np.zeros_like(h2)
    h2_shape = h2.shape
    for Q in range(1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0,trial,h2,q_list,Q,num_orb,num_electrons_up)
        K1s_K2s = get_k1s_k2s(q_list,Q)
        for K1,K2 in K1s_K2s:
            for g in range(h2_shape[2]):
                for r in range(num_orb):
                    avg_A_mat[(K1-1)*num_orb+r][(K2-1)*num_orb+r][g]=avg_A_vec_Q[g]
    return h2-avg_A_mat/num_electrons_up/2/num_k

def avg_A_Q(trial_0,trial,h2,q_list,q_selected,num_orb,num_electrons_up):
    '''
    It returns average of two-body Hamiltonian for a specified Q.
    '''
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    theta_full = theta(trial, trial)
    h2_shape = h2.shape[2]
    result = 1j*np.zeros(h2_shape)
    for K1,K2 in K1s_K2s:
        alpha = get_alpha_k1_k2(trial_0,h2,K1,K2,num_orb)
        result += contract('iiG->G',contract('nrG,rm->nmG',alpha,theta_full[(K2-1)*num_orb:K2*num_orb,(K1-1)*num_electrons_up:K1*num_electrons_up]))
    return 2*result

def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial,walker)))

def overlap(left_slater_det, right_slater_det):
    '''
    It computes the overlap between two Slater determinants.
    '''
    overlap_mat = np.dot(left_slater_det.transpose(),right_slater_det)
    #overlap_mat = np.array([right_slater_det[num_orb*k:num_orb*k+num_electrons_up] for k in range(num_k)]).reshape(num_k*num_electrons_up,num_k*num_electrons_up)
    return overlap_mat

def get_k1s_k2s(q_list,q_selected):
    '''
    It retuirns a list of tuples (k1s,k2s) corrsponding to q_selected.
    '''
    return list(zip(get_q_list(q_list,q_selected)[:,0],get_q_list(q_list,q_selected)[:,1]))

def get_q_list(q_list,q_selected):
    '''
    It returns q_list for a specicific momentum q_selected.
    '''
    return q_list[q_list[:,2]==q_selected]

def mean_field( h2, num_elec, num_band, num_k):
    mask = np.array(num_k * (num_elec * [True] + (num_band - num_elec) * [False]))
    m = np.sum(h2[mask,mask], axis=0)
    #m = np.einsum("iig->g", H2[mask][:,mask])
    #m = np.linalg.det( h2[:num_elec, : num_elec].T)
    return m

def get_alpha_k1_k2(trial_0,h2,k1_idx,k2_idx,num_orb):                                                               #####! it doesnt use mean field subtraction
    '''
    It returns alpha to be used as an intermediate object to compute one-body reduced density tensors.
    '''
    A_Q = get_A_k1_k2(h2,k1_idx,k2_idx,num_orb)
    result = np.einsum('ip,prG->irG',trial_0.T,A_Q)
    return result

def get_A_k1_k2(h2,k1_idx,k2_idx, num_orb):
    '''
    It returns the selected block of h2.
    '''
    return h2[(k1_idx-1)*num_orb:k1_idx*num_orb,(k2_idx-1)*num_orb:k2_idx*num_orb,:]

def H_1_mf(trial_0,trial,h2,h2_dagger,q_list,h1,num_k,num_orb,num_electrons_up):
    '''
    It returns one-body part of the Hamiltonian after mean-field subtraction.
    '''
    change = 1j*np.zeros_like(h1)
    for Q in range(1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0,trial,h2,q_list,Q,num_orb,num_electrons_up)
        avg_A_vec_Q_dagger = avg_A_Q(trial_0,trial,h2_dagger,q_list,Q,num_orb,num_electrons_up)
        K1s_K2s = get_k1s_k2s(q_list,Q)
        for K1,K2 in K1s_K2s:
            change[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb] = contract('rpG->rp',contract('G,rpG->rpG',avg_A_vec_Q_dagger,h2[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:])+contract('G,rpG->rpG',avg_A_vec_Q,h2_dagger[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:]))
    return h1+change/2

def gen_A_e_full(h2):
    '''
    It generates A_e.
    '''
    a_e = (h2 + np.einsum('ijG->jiG', h2.conj()))/2
    return a_e

def gen_A_o_full(h2):
    '''
    It generates A_o.
    '''
    a_o = (h2 - np.einsum('ijG->jiG', h2.conj()))*1j/2
    return a_o


def main(precision, backend):
    print()
    max_seed = np.iinfo(np.int32).max
    seed = np.random.randint(max_seed)
    if backend == "numpy":
        backend = NumpyBackend(seed)
    elif backend == "jax":
        backend = JaxBackend(seed)
    else:
        raise NotImplementedError(f"Selected {backend=} not implemented!")
    config = Configuration(
        num_walkers=10,
        num_kpoint=1,
        num_orbital=8,
        num_electron=4,
        num_g = 12039,
        singularity=0,
        propagator="S2",
        order_propagation=6,
        timestep=0.00075,
        comm=MPI.COMM_WORLD,
        precision=precision,
        backend=backend,
    )
    hamiltonian = Hamiltonian(
        one_body=obtain_H1(config),
        two_body=obtain_H2(config),
    )
    trial_det, walkers = initialize_determinant(config)
    hamiltonian.setup_energy_expressions(config, trial_det)

    np.random.seed(34841311)
    x_e_Q = np.random.randn(config.num_g, config.num_walkers)
    x_o_Q = np.random.randn(config.num_g, config.num_walkers)
    x_e_Qs = x_e_Q.astype(np.single)
    x_o_Qs = x_o_Q.astype(np.single)
    hamiltonian.test_random_field = np.concatenate((x_e_Qs, x_o_Qs), axis=0)

    expected_slater_det = np.load("slater_det.npy")
    expected_weights = np.load("weights.npy")
    new_walkers = propagate_walkers(config, trial_det, walkers, hamiltonian)
    check_det = np.allclose(new_walkers.slater_det, expected_slater_det, atol=1e-7)
    check_weight = np.allclose(new_walkers.weights, expected_weights)
    print("propagate_walkers", check_det, check_weight)
    print(new_walkers.slater_det.dtype, new_walkers.slater_det.__class__)
    print(new_walkers.weights.dtype, new_walkers.weights.__class__)

    E_hf = measure_energy(config, trial_det, walkers, hamiltonian)
    expected = 631.88947 - 2.8974954e-09j
    check_hf = np.isclose(E_hf, expected)
    np.random.seed(1887431)
    walkers.slater_det += 0.05 * np.random.rand(*walkers.slater_det.shape)
    E_random = measure_energy(config, trial_det, walkers, hamiltonian)
    expected = 631.8965 - 0.009482565j
    check_random = np.isclose(E_random, expected)
    print("measure_energy", check_hf, check_random)
    print(E_hf, E_hf.dtype, E_hf.__class__)
    print(E_random, E_random.dtype, E_random.__class__)


class Backend:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

class JaxBackend(Backend):
    def __init__(self, seed):
        super().__init__(jnp)
        self.block_diag = jax.scipy.linalg.block_diag
        self._key = jax.random.key(seed)

    def random_normal(self, shape, dtype):
        self._key, subkey = jax.random.split(self._key)
        return jax.random.normal(subkey, shape, dtype)


class NumpyBackend(Backend):
    def __init__(self, seed):
        super().__init__(np)
        self.block_diag = scipy.linalg.block_diag
        self._generator = np.random.default_rng(seed)

    def random_normal(self, shape, dtype):
        return self._generator.standard_normal(shape, dtype)


@dataclass
class Configuration:
    num_walkers: int  # number of walkers per core
    num_kpoint: int  # number of k-points to sample the Brillouin zone
    num_orbital: int  # size of the basis
    num_electron: int  # number of occupied states
    num_g: int  # number of points in the factorization
    singularity: float  # singularity used for the G=0 component (fsg)
    propagator: str  # select which method to use to propagate in time
    order_propagation: int  # number of times the Hamiltonian is applied in propagator
    timestep: float  # imaginary time passing between two steps
    comm: MPI.Comm  # communicator over which the walkers are distributed
    precision: str  # must be either single or double
    backend: types.ModuleType  # module used to execute numpy-like operations

    @property
    def float_type(self):
        if self.precision == "single":
            return self.backend.single
        elif self.precision == "double":
            return self.backend.double
        else:
            raise NotImplementedError(self._precision_error_message())

    @property
    def complex_type(self):
        if self.precision == "single":
            return self.backend.csingle
        elif self.precision == "double":
            return self.backend.cdouble
        else:
            raise NotImplementedError(self._precision_error_message())

    def _precision_error_message(self):
        return f"Specified precision '{self.precision}' not implemented."


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
    test_random_field: np.typing.ArrayLike = None  # a random field used for testing

    def setup_energy_expressions(self, config, trial_det):
        shape_theta = (
            config.num_walkers,
            config.num_orbital * config.num_kpoint,
            config.num_electron * config.num_kpoint,
        )
        h1_trial = trial_det.T @ self.one_body
        alpha = contract("pi, prg -> irg", trial_det, self.two_body)
        alpha_T = contract("pi, rpg -> irg", trial_det, self.two_body.conj())
        self._one_body_expression = contract_expression(
            "ip, wpi -> w", h1_trial, shape_theta, constants=[0], optimize="greedy"
        )
        args = (shape_theta, alpha, shape_theta, alpha_T)
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

        ql = []
        for i in range(1,config.num_kpoint+1):
            for j in range(1,config.num_kpoint+1):
                for k in range(1,config.num_kpoint+1):
                    if abs(i-j)==k-1:
                        ql.append([i,j,k])
        ql = np.array(ql)

        h_mf = self.compute_mean_field_one_body(config, trial_det, ql)
        h_sic = -contract('ijG, kjG -> ik', self.two_body, self.two_body.conj()) / 2
        h1 = self.one_body + h_mf + h_sic
        self._h1 = -h1 * config.timestep
        self._exp_h1 = expm(-h1 * config.timestep)
        self._exp_h1_half = expm(-0.5 * h1 * config.timestep)

        two_body = self.compute_mean_field_two_body(config, trial_det, ql)
        alpha = contract("pi, prg -> irg", trial_det, two_body)
        self._sqrt_tau = config.backend.sqrt(config.timestep).astype(config.float_type)
        self._force_bias_expression = contract_expression(
            'wri, irg -> gw',
            (config.num_walkers, config.num_orbital * config.num_kpoint, config.num_electron * config.num_kpoint),
            alpha,
            constants=[1],
            optimize='greedy'
        )
        self._auxiliary_field = contract_expression(
            'ijg, gw -> ijw',
            two_body,
            (2 * config.num_g, config.num_walkers),
            constants=[0],
            optimize='greedy'
        )

    def compute_one_body(self, theta):
        return 2 * self._one_body_expression(theta)

    def compute_hartree(self, theta):
        return 2 * self._hartree_expression(theta, theta)

    def compute_exchange(self, theta):
        return -self._exchange_expression(theta, theta) - self._singularity_correction

    def create_auxiliary_field(self, config, theta):
        random_field = self.create_random_field(config)
        force_bias = -2j * self._sqrt_tau * self._force_bias_expression(theta)
        # Boundary condition for rare events based on: https://doi.org/10.1103/PhysRevB.80.214116
        force_bias = config.backend.where(abs(force_bias) > 1, 1.0, force_bias)
        arg = contract("gw, gw -> w", random_field - 0.5 * force_bias, force_bias)
        field = 1j * self._sqrt_tau * self._auxiliary_field(random_field - force_bias)
        return field, np.exp(arg)

    def create_random_field(self, config):
        if self.test_random_field is None:
            shape = (2 * config.num_g, config.num_walkers)
            return config.backend.random_normal(shape, config.float_type)
        else:
            return self.test_random_field

    def compute_mean_field_one_body(self, config, trial_det, ql):
        # interface to legacy code
        h2_t = np.einsum('prG->rpG', self.two_body.conj())
        h1_legacy = H_1_mf(trial_det,trial_det,self.two_body,h2_t,ql,self.one_body,config.num_kpoint,config.num_orbital,config.num_electron)
        result = h1_legacy - self.one_body
        return config.backend.array(result, dtype=config.complex_type)

    def compute_mean_field_two_body(self, config, trial_det, ql):
        # interface to legacy code
        h2_af_MF_sub = A_af_MF_sub(trial_det,trial_det,self.two_body,ql,config.num_kpoint,config.num_orbital,config.num_electron)
        two_body_e = gen_A_e_full(h2_af_MF_sub)
        two_body_o = gen_A_o_full(h2_af_MF_sub)
        result = np.concatenate((two_body_e, two_body_o), axis=-1)
        return config.backend.array(result, dtype=config.complex_type)

    @property
    def exp_h0(self):
        return 1

    @property
    def h1(self):
        return self._h1

    @property
    def exp_h1(self):
        return self._exp_h1

    @property
    def exp_h1_half(self):
        return self._exp_h1_half


def obtain_H1(config):
    """
    Reshape H1 (H1.npy shape is num_orb, num_orb, num_k
    we implement the k_points to the H1)
    """
    input_h1 = config.backend.load("H1.npy").astype(config.complex_type)
    output_h1 = config.backend.moveaxis(input_h1, -1, 0)
    return config.backend.block_diag(*output_h1)


def obtain_H2(config):
    input_h2 = config.backend.load("H2.npy").astype(config.complex_type)
    return config.backend.moveaxis(input_h2, 0, -1)


def initialize_determinant(config):
    shape = (config.num_orbital, config.num_electron)
    trial_det = config.backend.eye(*shape, dtype=config.float_type)
    slater_det = config.backend.array(config.num_walkers * [trial_det])
    walkers = Walkers(
        slater_det=slater_det.astype(config.complex_type),
        weights=config.backend.ones(config.num_walkers, dtype=config.complex_type),
    )
    return trial_det, walkers


def measure_energy(config, trial, walkers, hamiltonian):
    # compute energies locally
    theta = biorthogonalize(config.backend, trial, walkers.slater_det)
    energy_one_body = hamiltonian.compute_one_body(theta)
    energy_hartree = hamiltonian.compute_hartree(theta)
    energy_exchange = hamiltonian.compute_exchange(theta)
    energy = (energy_one_body + energy_hartree + energy_exchange) / config.num_kpoint
    weighted_energy = energy @ walkers.weights

    # average over all ranks
    sum_weights = config.backend.sum(walkers.weights)
    weighted_energy_global = config.comm.allreduce(weighted_energy)
    sum_weights_global = config.comm.allreduce(sum_weights)
    return weighted_energy_global / sum_weights_global


def propagate_walkers(config, trial, walkers, hamiltonian):
    new_walkers = Walkers(np.zeros_like(walkers.slater_det), np.zeros_like(walkers.weights))
    theta = biorthogonalize(config.backend, trial, walkers.slater_det)
    h2, importance = hamiltonian.create_auxiliary_field(config, theta)
    print('h_2 type', h2.dtype, h2.__class__)

    if config.propagator == "Taylor":
        h = hamiltonian.h1 + h2
        full_step_with_h1_and_h2 = apply_taylor(config, h, walkers.slater_det)
        new_walkers.slater_det = hamiltonian.exp_h0 * full_step_with_h1_and_h2

    elif config.propagator == "S1":
        full_step_with_h2 = apply_taylor(config, h2, half_step_with_h1)
        full_step_with_h1 = hamiltonian.exp_h1 @ full_step_with_h2
        new_walkers.slater_det = hamiltonian.exp_h0 * full_step_with_h1

    elif config.propagator == "S2":
        half_step_with_h1 = hamiltonian.exp_h1_half @ walkers.slater_det
        full_step_with_h2 = apply_taylor(config, h2, half_step_with_h1)
        half_step_with_h1 = hamiltonian.exp_h1_half @ full_step_with_h2
        new_walkers.slater_det = hamiltonian.exp_h0 * half_step_with_h1

    else:
        raise ValueError("Invalid method selected. Choose from 'taylor', 'S1', or 'S2'.")

    new_overlap = project_trial(config.backend, trial, new_walkers.slater_det)
    old_overlap = project_trial(config.backend, trial, walkers.slater_det)
    overlap_ratio = new_overlap / old_overlap
    cos_alpha = np.cos(config.backend.angle(overlap_ratio))
    factor = np.abs(overlap_ratio * importance) * np.maximum(0, cos_alpha)
    new_walkers.weights = factor * walkers.weights

    return new_walkers


def apply_taylor(config, matrix, slater_det):
    result = slater_det.copy()
    addend = slater_det
    for i in range(config.order_propagation):
        # addend = contract("pq, qi -> pi", matrix , addend) / (i + 1)
        addend = contract("pqw, wqi -> wpi", matrix , addend) / (i + 1)
        result += addend
    return result

def biorthogonalize(backend, trial, slater_det):
    inverse_overlap = backend.linalg.inv(trial.T @ slater_det)
    return contract("wpi, wij -> wpj", slater_det, inverse_overlap)

def project_trial(backend, trial, slater_det):
    overlap = trial.T @ slater_det
    return backend.linalg.det(overlap)**2

if __name__ == "__main__":
    main("single", "numpy")
    main("double", "numpy")
    main("single", "jax")
    main("double", "jax")
