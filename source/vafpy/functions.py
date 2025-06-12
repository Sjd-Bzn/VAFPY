from dataclasses import dataclass
import types
from time import time
import numpy as np
import cupy as cp
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

def A_af_MF_sub(trial_0,trial,h2,q_list,num_k,num_orb,num_electrons_up, config):
    '''
    It returns two body Hamiltonian after mean-field subtraction.
    '''
    avg_A_mat = np.zeros_like(h2)
    h2_shape = h2.shape
    for Q in range(1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0,trial,h2,q_list,Q,num_orb,num_electrons_up, config)
        K1s_K2s = get_k1s_k2s(q_list,Q)
        for K1,K2 in K1s_K2s:
            for g in range(h2_shape[2]):
                for r in range(num_orb):
                    avg_A_mat[(K1-1)*num_orb+r][(K2-1)*num_orb+r][g]=avg_A_vec_Q[g]
    return h2-avg_A_mat/num_electrons_up/2/num_k

def avg_A_Q(trial_0,trial,h2,q_list,q_selected,num_orb,num_electrons_up, config):
    '''
    It returns average of two-body Hamiltonian for a specified Q.
    '''
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    theta_full = theta(trial, trial)
    h2_shape = h2.shape[2]
    result = 1j*config.backend.zeros(h2_shape)
    #result = 1j*cp.zeros(h2_shape)
    for K1,K2 in K1s_K2s:
        alpha = get_alpha_k1_k2(trial_0,h2,K1,K2,num_orb, config)
        print(f"{result.__class__=} {alpha.__class__=}")
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

def get_alpha_k1_k2(trial_0,h2,k1_idx,k2_idx,num_orb, config):                                                               #####! it doesnt use mean field subtraction
    '''
    It returns alpha to be used as an intermediate object to compute one-body reduced density tensors.
    '''
    A_Q = get_A_k1_k2(h2,k1_idx,k2_idx,num_orb)
    result = config.backend.einsum('ip,prG->irG',trial_0.T,A_Q)
    return result

def get_A_k1_k2(h2,k1_idx,k2_idx, num_orb):
    '''
    It returns the selected block of h2.
    '''
    return h2[(k1_idx-1)*num_orb:k1_idx*num_orb,(k2_idx-1)*num_orb:k2_idx*num_orb,:]

def H_1_mf(trial_0,trial,h2,h2_dagger,q_list,h1,num_k,num_orb,num_electrons_up, config):
    '''
    It returns one-body part of the Hamiltonian after mean-field subtraction.
    '''
    change = 1j*np.zeros_like(h1)
    for Q in range(1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0,trial,h2,q_list,Q,num_orb,num_electrons_up, config)
        avg_A_vec_Q_dagger = avg_A_Q(trial_0,trial,h2_dagger,q_list,Q,num_orb,num_electrons_up, config)
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
    #print()
   # max_seed = np.iinfo(np.int32).max
   # seed = np.random.randint(max_seed)
    if backend == "numpy":
        backend = NumpyBackend(seed)
    elif backend == "jax":
        backend = JaxBackend(seed)
    elif backend == "cupy":
        backend = CupyBackend(seed)
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

    #np.random.seed(34841311)
    #x_e_Q = np.random.randn(config.num_g, config.num_walkers)
    #x_o_Q = np.random.randn(config.num_g, config.num_walkers)
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
    #np.random.seed(1887431)
    #random_change = config.backend.array(np.random.rand(*walkers.slater_det.shape))
    walkers.slater_det += 0.05 * random_change.astype(config.complex_type)
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
        #self.block_diag = jax.scipy.linalg.block_diag
        self._key = jax.random.key(seed)
        #self.expm = jax.scipy.linalg.expm


    def block_diag(self, *matrices):
        matrix_copy = [np.asarray(matrix) for matrix in matrices]
        matrix = scipy.linalg.block_diag(*matrix_copy)
        return jnp.array(matrix)

    def expm(self, matrix):
        result = scipy.linalg.expm(matrix)
        return jnp.array(result)

    def random_normal(self, shape, dtype):
        self._key, subkey = jax.random.split(self._key)
        return jax.random.normal(subkey, shape, dtype)

    def random_uniform(self, shape, dtype):
        self._key, subkey = jax.random.split(self._key)
        return jax.random.uniform(subkey, shape, dtype=dtype)



class NumpyBackend(Backend):
    def __init__(self, seed):
        super().__init__(np)
        self.block_diag = scipy.linalg.block_diag
        self.expm = scipy.linalg.expm
        self._generator_1 = np.random.default_rng(seed)
        #self._generator_2 = np.random.default_rng(seed)

    def random_normal(self, shape, dtype):
        #a = self._generator_1.standard_normal(shape, dtype)
        #b = self._generator_2.standard_normal(shape, dtype)
        #print(np.allclose(a, b))
        return self._generator_1.standard_normal(shape).astype(dtype)

    def random_uniform(self, shape, dtype):
        return self._generator_1.uniform(size=shape).astype(dtype)


class CupyBackend(Backend):
    def __init__(self, seed):
        super().__init__(cp)
        self._generator = cp.random.default_rng(seed)

    def block_diag(self, *matrices):
        matrix_copies = (matrix.get() for matrix in matrices)
        matrix = scipy.linalg.block_diag(*matrix_copies)
        return cp.array(matrix)

    def expm(self, matrix):
        exp_matrix = scipy.linalg.expm(matrix.get())
        return cp.array(exp_matrix)

    def random_normal(self, shape, dtype):
        return self._generator.standard_normal(shape).astype(dtype)

    def random_uniform(self, shape, dtype):
        return self._generator.uniform(size=shape).astype(dtype)


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
        if self.precision == "Single":
            return self.backend.single
        elif self.precision == "Double":
            return self.backend.double
        else:
            raise NotImplementedError(self._precision_error_message())

    @property
    def complex_type(self):
        if self.precision == "Single":
            return self.backend.csingle
        elif self.precision == "Double":
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
        #print("h_mf", h_mf)
        #print("h_sic", h_sic)
        self._h1 = -h1 * config.timestep
        self._exp_h1 = config.backend.expm(-h1 * config.timestep)
        self._exp_h1_half = config.backend.expm(-0.5 * h1 * config.timestep)
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
        a = 2 * self._hartree_expression(theta, theta)
    #    print('a',a)
        return a
    def compute_exchange(self, theta):
        return -self._exchange_expression(theta, theta) #- self._singularity_correction

    def create_auxiliary_field(self, config, theta):
#        st_time = time()
        random_field = self.create_random_field(config)
        force_bias = -2j * self._sqrt_tau * self._force_bias_expression(theta)
        # Boundary condition for rare events based on: https://doi.org/10.1103/PhysRevB.80.214116
        force_bias = config.backend.where(abs(force_bias) > 1, 0.0, force_bias)
        arg = contract("gw, gw -> w", random_field - 0.5 * force_bias, force_bias)
        field = 1j * self._sqrt_tau * self._auxiliary_field(random_field - force_bias)
 #       fi_time = time()
  #      print("Aux time" ,st_time - fi_time)
        return field, config.backend.exp(arg)

    def create_random_field(self, config):
        if self.test_random_field is None:
            shape = (2 * config.num_g, config.num_walkers)
            return config.backend.random_normal(shape, config.float_type)
        else:
            return self.test_random_field

    def compute_mean_field_one_body(self, config, trial_det, ql):
        # interface to legacy code
        h2_t = np.einsum('prG->rpG', self.two_body.conj())
        num_orb = config.num_orbital
        alpha = contract('pi,prG->irG', trial_det, self.two_body)
        avg_A_vec_Q = 2 * contract('irG,ri->G', alpha, trial_det)
        alpha = contract('pi,rpG->irG', trial_det, self.two_body.conj())
        avg_A_vec_Q_dagger = 2 * contract('irG,ri->G', alpha, trial_det)
        change = contract('G,rpG->rp',avg_A_vec_Q_dagger,self.two_body[:,:,:]) + contract('G,rpG->rp',avg_A_vec_Q,h2_t[:,:,:])

        return config.backend.array(change/2, dtype=config.complex_type)

    def compute_mean_field_two_body(self, config, trial_det, ql):
        # interface to legacy code
        h2_af_MF_sub = A_af_MF_sub(trial_det,trial_det,self.two_body,ql,config.num_kpoint,config.num_orbital,config.num_electron, config)
        two_body_e = gen_A_e_full(h2_af_MF_sub)
        two_body_o = gen_A_o_full(h2_af_MF_sub)
        result = np.concatenate((two_body_e, two_body_o), axis=-1)
        return config.backend.array(result, dtype=config.complex_type)


    def exp_h0( config, theta):
        a =  2 * compute_hartreei( theta) / (2*config.num_electron)
        print("a",a)
        b = np.exp(config.timestep*a)
        print("b",b)
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
    input_h2 = config.backend.load("H2_zip.npy").astype(config.complex_type)
    return input_h2

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
    return weighted_energy_global / sum_weights_global, weighted_energy_global, sum_weights_global

def measure_hartree(config, trial, walkers, hamiltonian):
    # compute energies locally
    theta = biorthogonalize(config.backend, trial, walkers.slater_det)
    energy_hartree = hamiltonian.compute_hartree(theta)
    energy = ( energy_hartree ) / config.num_kpoint
    weighted_energy = energy @ walkers.weights

    # average over all ranks
    sum_weights = config.backend.sum(walkers.weights)
    weighted_energy_global = config.comm.allreduce(weighted_energy)
    sum_weights_global = config.comm.allreduce(sum_weights)
    return weighted_energy_global / sum_weights_global


#tot_aux_time = 0
def propagate_walkers(config, trial, walkers, hamiltonian, h_0, e_0):
    new_walkers = Walkers(np.zeros_like(walkers.slater_det), np.zeros_like(walkers.weights))
    theta = biorthogonalize(config.backend, trial, walkers.slater_det)
    h2, importance = hamiltonian.create_auxiliary_field(config, theta)
    #tot_aux_time += au_time
    num_rare_event = 0
#    print('h_2 type', h2.dtype, h2.__class__)

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
        #print("half_step_with_h1" ,half_step_with_h1.shape)
        full_step_with_h2 = apply_taylor(config, h2, half_step_with_h1)
        half_step_with_h1 = hamiltonian.exp_h1_half @ full_step_with_h2
        new_walkers.slater_det = half_step_with_h1 * h_0
        #print("walker", half_step_with_h1.shape)
        #print("h0", hamiltonian.exp_h0.shape)
    else:
        raise ValueError("Invalid method selected. Choose from 'taylor', 'S1', or 'S2'.")

    new_overlap = project_trial(config.backend, trial, new_walkers.slater_det)
    old_overlap = project_trial(config.backend, trial, walkers.slater_det)
    overlap_ratio = new_overlap / old_overlap
    cos_alpha = config.backend.cos(config.backend.angle(overlap_ratio))
    factor = abs(overlap_ratio * importance * (config.backend.exp(config.timestep*e_0)))

    factor = config.backend.where(factor < 10, factor, 0)  # Fixes ambiguous truth value issue
    #print("factor", np.mean(factor))
    num_rare_event += config.backend.sum(factor == 0)  # Count number of events where factor was reset
    #if np.all(np.abs(factor) < 10):
   #     factor = factor
   # else:
   #     factor = 0
   #     num_rare_event += 1
    a = abs(factor)* config.backend.maximum(0, cos_alpha)
    new_walkers.weights = a * walkers.weights

    return new_walkers, num_rare_event


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


def rebalance_comb(weights):
    """
    Rebalances weights using systematic (stratified) resampling.
    Returns an array of new walker indices (length = N).
    """
    N = len(weights)
    c = config.backend.cumsum(weights.real)
    W = c[-1]            # total weight
    # random offset in [0, W/N)
    r = config.backend.random_uniform(0, W/N)

    new_indices = config.backend.zeros(N, dtype=config.int_type)

    # pointer for searching in c
    i = 0
    for k in range(N):
        U = r + k*(W/N)
        # Move i forward until c[i] >= U
        while U > c[i]:
            i += 1
        new_indices[k] = i

    return new_indices






def rebalance_global(comm, walkers_mats_up, walkers_weights, config):
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_n, N_orb, N_elec = walkers_mats_up.shape
    total_n = local_n * size

    # Step 1: Gather weights globally
    all_weights = None
    if rank == 0:
        all_weights = config.backend.empty(total_n, dtype=config.complex_type)
    comm.Gather(walkers_weights, all_weights, root=0)

    # Step 2: Normalize weights and determine global instances (rank 0 only)
    instances = None
    if rank == 0:
        norm_factor = total_n / config.backend.sum(all_weights.real)
        norm_weights = all_weights.real * norm_factor

        # Random bias (shift)
        bias = -config.backend.random_uniform((), config.float_type)


        # Determine how many copies (instances) each walker has after resampling
        instances = config.backend.zeros(total_n, dtype=config.int_type)     ########int is required not float
        cumulative_sum = bias
        previous = 0
        for i in range(total_n):
            cumulative_sum += norm_weights[i]
            current = int(config.backend.ceil(cumulative_sum))
            instances[i] = current - previous
            previous = current

    # Step 3: Broadcast instances to all ranks
    if rank != 0:
        instances = config.backend.empty(total_n, dtype=config.int_type)     ########int
    comm.Bcast(instances, root=0)

    # Step 4: Determine global mapping of walkers after rebalancing
    new_total_walkers = config.backend.sum(instances)
    map_indices = config.backend.empty(new_total_walkers, dtype=config.int_type)     ######int
    count = 0
    for idx, num_instances in enumerate(instances):
        for _ in range(num_instances):
            map_indices[count] = idx
            count += 1

    # Step 5: Distribute resampled walkers evenly back to ranks
    assert new_total_walkers == total_n, "Walker count mismatch after rebalancing."

    # Scatter the new walkers based on the global mapping
    # Gather all walkers to rank 0 first
    all_mats_up = None
    if rank == 0:
        all_mats_up = config.backend.empty((total_n, N_orb, N_elec), dtype=config.complex_type)
    comm.Gather(walkers_mats_up, all_mats_up, root=0)

    # Rearrange walkers according to the resampled indices
    resampled_mats_up = None
    if rank == 0:
        resampled_mats_up = all_mats_up[map_indices]

    # Scatter evenly back to ranks
    new_mats_up = config.backend.empty((local_n, N_orb, N_elec), dtype=config.complex_type)
    comm.Scatter(resampled_mats_up, new_mats_up, root=0)

    # Step 6: Reset weights uniformly
    new_weights = config.backend.ones(local_n, dtype=config.complex_type)

    return new_mats_up, new_weights

#def rebalance_global(comm, walkers_mats_up, walkers_weights, config):
#    backend = config.backend
#    rank = comm.Get_rank()
#    size = comm.Get_size()
#
#    local_n, N_orb, N_elec = walkers_mats_up.shape
#    total_n = local_n * size
#
#    # Step 1: Gather weights globally
#    all_weights = None
#    if rank == 0:
#        all_weights = backend.empty(total_n, dtype=config.complex_type)
#    # For MPI, ensure walkers_weights is on host (NumPy)
#    comm.Gather(
#        walkers_weights if backend.__name__ == 'numpy' else backend.asarray(walkers_weights),
#        all_weights,
#        root=0
#    )
#
#    # Step 2: Normalize weights and determine global instances (rank 0 only)
#    instances = None
#    if rank == 0:
#        total_weight = backend.sum(all_weights.real)
#        norm_factor = total_n / total_weight
#        norm_weights = all_weights.real * norm_factor
#
#        # Random bias (shift)
#        bias = -config.backend.random_uniform((), config.float_type)
#
#        # Determine how many copies (instances) each walker has after resampling
#        instances = backend.zeros(total_n, dtype=int)
#
#        cumulative_sum = bias
#        previous = 0
#        instances_list = []  # We cannot dynamically index arrays easily in JAX, so use Python list
#
#        norm_weights_host = backend.device_get(norm_weights) if hasattr(backend, 'device_get') else norm_weights
#        for w in norm_weights_host:
#            cumulative_sum += w
#            current = int(backend.ceil(cumulative_sum))
#            instances_list.append(current - previous)
#            previous = current
#
#        instances = backend.array(instances_list, dtype=int)
#
#    # Step 3: Broadcast instances to all ranks
#    if rank != 0:
#        instances = backend.empty(total_n, dtype=int)
#    comm.Bcast(
#        instances if backend.__name__ == 'numpy' else backend.asarray(instances),
#        root=0
#    )
#
#    # Step 4: Determine global mapping of walkers after rebalancing
#    new_total_walkers = backend.sum(instances).item()  # get scalar
#    map_indices = backend.empty(new_total_walkers, dtype=int)
#
#    count = 0
#    indices_list = []
#    instances_host = backend.device_get(instances) if hasattr(backend, 'device_get') else instances
#    for idx, num_instances in enumerate(instances_host):
#        for _ in range(num_instances):
#            indices_list.append(idx)
#
#    map_indices = backend.array(indices_list, dtype=int)
#
#    # Step 5: Distribute resampled walkers evenly back to ranks
#    assert new_total_walkers == total_n, "Walker count mismatch after rebalancing."
#
#    all_mats_up = None
#    if rank == 0:
#        all_mats_up = backend.empty((total_n, N_orb, N_elec), dtype=config.complex_type)
#    comm.Gather(
#        walkers_mats_up if backend.__name__ == 'numpy' else backend.asarray(walkers_mats_up),
#        all_mats_up,
#        root=0
#    )
#
#    resampled_mats_up = None
#    if rank == 0:
#        resampled_mats_up = all_mats_up[map_indices]
#
#    new_mats_up = backend.empty((local_n, N_orb, N_elec), dtype=config.complex_type)
#    comm.Scatter(
#        resampled_mats_up if backend.__name__ == 'numpy' else backend.asarray(resampled_mats_up),
#        new_mats_up,
#        root=0
#    )
#
#    # Step 6: Reset weights uniformly
#    new_weights = backend.ones(local_n, dtype=config.complex_type)
#
#    return new_mats_up, new_weights
#




def reortho_qr(walker_matrix, config):
    '''
    It uses QR decomposition to reorthogonalize the orbitals in a single walker.
    '''
    Q, R = config.backend.linalg.qr(walker_matrix)
    return Q

def init_walkers_weights(n_walkers, config):
    '''
    It initializes the weights.
    '''
    return config.backend.ones(n_walkers, dtype=config.complex_type)


def blockAverage(datastream, block_divisor):
    Nobs = len(datastream)
    N = block_divisor
    minBlockSize = 1;
    maxBlockSize = int(Nobs/N);
    NumBlocks = maxBlockSize - minBlockSize
    blockMean = np.zeros(NumBlocks)
    blockVar = np.zeros(NumBlocks)
    blockCtr = 0
    for blockSize in range(minBlockSize, maxBlockSize):
        Nblock = int(Nobs/blockSize)
        obsProp = np.zeros(Nblock)
        for i in range(1,Nblock+1):
            ibeg = (i-1) * blockSize
            iend =  ibeg + blockSize
            obsProp[i-1] = np.mean(datastream[ibeg:iend])
        blockMean[blockCtr] = np.mean(obsProp)
        blockVar[blockCtr] = (np.var(obsProp)/(Nblock - 1))
        blockCtr += 1
    v = np.arange(minBlockSize,maxBlockSize)
    return v, blockVar, blockMean




if __name__ == "__main__":
    main("single", "numpy")
    main("double", "numpy")
    main("single", "jax")
    main("double", "jax")
