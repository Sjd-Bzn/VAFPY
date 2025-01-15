from dataclasses import dataclass
import types

import numpy as np
import jax
import jax.numpy as jnp
import scipy
from mpi4py import MPI
from opt_einsum import contract, contract_expression

from scipy.linalg import expm
from cmath import phase

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


def propagate_walkers(config, trial,walkers,h_0,h_1,d_tau,e_0,h1_exp_half,propagator, x_e_Q, x_o_Q,num_k,num_orb,SQRT_DTAU,expr_fb_e,expr_fb_o,NUM_WALKERS,order_trunc,expr_h2_e,expr_h2_o):
    new_walkers = Walkers(np.zeros_like(walkers.slater_det), np.zeros_like(walkers.weights))
    theta = biorthogonalize(config.backend, trial, walkers.slater_det)

    fb_e_Q = -2j*SQRT_DTAU*expr_fb_e(theta)
    fb_o_Q = -2j*SQRT_DTAU*expr_fb_o(theta)

    ####Boundary condition for rare events  based on: https://doi.org/10.1103/PhysRevB.80.214116

    fb_e_Q[abs(fb_e_Q)>1] = 1.0
    fb_o_Q[abs(fb_o_Q)>1] = 1.0

    h_2 = expr_h2_e(x_e_Q-fb_e_Q.T)+expr_h2_o(x_o_Q-fb_o_Q.T)
    print('h_2 type', h_2.dtype)
    for i in range(0,NUM_WALKERS):
        if propagator == 'Old':
            h= h_1 + SQRT_DTAU * 1j *h_2 [:,:,i]
            addend = walkers.slater_det[i]
            for j in range(order_trunc+1):
                new_walkers.slater_det[i] += addend
                addend = h@addend/(j + 1)

########       Different propagators Taylor, S1, and S2

        elif propagator == 'Taylor':
            prop_taylor = np.exp(d_tau * (-h_0 + e_0)) * exp_Taylor(h, num_k, num_orb, order_trunc)
            new_walkers.slater_det[i] = prop_taylor @ walkers.slater_det[i]

        elif propagator == 'S1':

            prop_S1 = np.exp(d_tau * (-h_0 + e_0)) * expm(-d_tau * h_1) * exp_Taylor(SQRT_DTAU * 1j * h_2[:, :, i], num_k, num_orb, order_trunc)
            new_walkers.slater_det[i] = prop_S1 @ walkers.slater_det[i]

        elif propagator == 'S2':
            prop_S2 = h1_exp_half@exp_Taylor(SQRT_DTAU*1j*h_2[:, :, i],num_k,num_orb,order_trunc)@h1_exp_half
            new_walkers.slater_det[i] = prop_S2 @ walkers.slater_det[i]
        else:
            raise ValueError("Invalid method selected. Choose from 'taylor', 'S1', or 'S2'.")

        ovrlap_ratio = np.linalg.det(overlap(trial,new_walkers.slater_det[i]))**2 / np.linalg.det(overlap(trial,walkers.slater_det[i]))**2
        alpha = phase(ovrlap_ratio)

####new_weight

        new_walkers.weights[i] = abs(ovrlap_ratio*(np.exp( np.dot(x_e_Q[:,i],fb_e_Q[i])-np.dot(fb_e_Q[i],fb_e_Q[i]/2))*np.exp(np.dot(x_o_Q[:,i], fb_o_Q[i])-np.dot(fb_o_Q[i],fb_o_Q[i]/2))))* max(0,np.cos(alpha))*walkers.weights[i]
    return new_walkers

def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial,walker)))


def exp_Taylor(mat,num_k,num_orb,order_trunc):
    OUT = np.eye(num_orb*num_k,dtype = 'complex128')
    C = np.eye(num_orb*num_k)
    for i in range (0,order_trunc):
      C = mat@C/(i+1)
      OUT += C
    return OUT

def overlap(left_slater_det, right_slater_det):
    '''
    It computes the overlap between two Slater determinants.
    '''
    overlap_mat = np.dot(left_slater_det.transpose(),right_slater_det)
    #overlap_mat = np.array([right_slater_det[num_orb*k:num_orb*k+num_electrons_up] for k in range(num_k)]).reshape(num_k*num_electrons_up,num_k*num_electrons_up)
    return overlap_mat

def main(precision, backend):
    if backend == "numpy":
        backend = np
        backend.block_diag = scipy.linalg.block_diag
    elif backend == "jax":
        backend = jnp
        backend.block_diag = jax.scipy.linalg.block_diag
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

    SQRT_DTAU = np.sqrt(config.timestep)

    h2_t = np.einsum('prG->rpG', hamiltonian.two_body.conj())

    ql = []
    for i in range(1,config.num_kpoint+1):
        for j in range(1,config.num_kpoint+1):
            for k in range(1,config.num_kpoint+1):
                if abs(i-j)==k-1:
                    ql.append([i,j,k])
    ql = np.array(ql)

    np.random.seed(34841311)
    x_e_Q = np.random.randn(config.num_g, config.num_walkers)
    x_o_Q = np.random.randn(config.num_g, config.num_walkers)
    x_e_Qs = x_e_Q.astype(np.single)
    x_o_Qs = x_o_Q.astype(np.single)

    hamil_MF = HAMILTONIAN_MF
    h2_af_MF_sub = A_af_MF_sub(trial_det,trial_det,hamiltonian.two_body,ql,config.num_kpoint,config.num_orbital,config.num_electron)
    L_0 = mean_field(hamiltonian.two_body, config.num_electron, config.num_orbital, config.num_kpoint)
    H_zero= np.einsum("g, g->", L_0, L_0 )/2/2/config.num_kpoint
    hamil_MF.zero_body= H_zero
    hamil_MF.one_body = H_1_mf(trial_det,trial_det,hamiltonian.two_body,h2_t,ql,hamiltonian.one_body,config.num_kpoint,config.num_orbital,config.num_electron)
    hamil_MF.two_body_e = gen_A_e_full(h2_af_MF_sub)
    hamil_MF.two_body_o = gen_A_o_full(h2_af_MF_sub)
    ALPHA_E = contract('ip,prG->irG',trial_det.T,hamil_MF.two_body_e)
    ALPHA_O = contract('ip,prG->irG',trial_det.T,hamil_MF.two_body_o)
    ALPHA_E_s = ALPHA_E.astype(np.complex64)
    ALPHA_O_s = ALPHA_O.astype(np.complex64)
    expr_fb_e = contract_expression('Nri,irG->NG',(config.num_walkers,config.num_orbital*config.num_kpoint,config.num_electron*config.num_kpoint),ALPHA_E_s,constants=[1],optimize='greedy')
    expr_fb_o = contract_expression('Nri,irG->NG',(config.num_walkers,config.num_orbital*config.num_kpoint,config.num_electron*config.num_kpoint),ALPHA_O_s,constants=[1],optimize='greedy')
    expr_h2_e = contract_expression('ijG,GN->ijN',hamil_MF.two_body_e.astype(np.complex64),(config.num_g,config.num_walkers),constants=[0],optimize='greedy')
    expr_h2_o = contract_expression('ijG,GN->ijN',hamil_MF.two_body_o.astype(np.complex64),(config.num_g,config.num_walkers),constants=[0],optimize='greedy')

    h_self = -contract('ijG,jkG->ik',hamiltonian.two_body,h2_t)/2#/num_k
    H_1_self = -config.timestep * (hamil_MF.one_body+h_self)
    H1_self_half_exp = expm(H_1_self/2).astype(np.complex64)

    walkers_single = Walkers(walkers.slater_det.astype(np.csingle), walkers.weights.astype(np.csingle))
    walkers_double = Walkers(walkers.slater_det.astype(np.cdouble), walkers.weights.astype(np.cdouble))

    expected_slater_det = np.load("slater_det.npy")
    expected_weights = np.load("weights.npy")
    walkers_single = propagate_walkers(config, trial_det, walkers_single,0,hamiltonian.one_body,config.timestep,0,H1_self_half_exp,config.propagator,x_e_Qs,x_o_Qs,config.num_kpoint,config.num_orbital,SQRT_DTAU,expr_fb_e,expr_fb_o,config.num_walkers,config.order_propagation,expr_h2_e,expr_h2_o)
    print("single", np.allclose(walkers_single.slater_det, expected_slater_det), np.allclose(walkers_single.weights, expected_weights))

    walkers_double = propagate_walkers(config, trial_det,walkers_double,0,hamiltonian.one_body,config.timestep,0,H1_self_half_exp,config.propagator,x_e_Q,x_o_Q,config.num_kpoint,config.num_orbital,SQRT_DTAU,expr_fb_e,expr_fb_o,config.num_walkers,config.order_propagation,expr_h2_e,expr_h2_o)
    print("double", np.allclose(walkers_double.slater_det, expected_slater_det), np.allclose(walkers_double.weights, expected_weights))

    E = measure_energy(config, trial_det, walkers, hamiltonian)
    print(E.dtype, E.__class__)
    expected = 631.88947 - 2.8974954e-09j
    print(E, np.isclose(E, expected))
    np.random.seed(1887431)
    walkers.slater_det += 0.05 * np.random.rand(*walkers.slater_det.shape)
    E = measure_energy(config, trial_det, walkers, hamiltonian)
    expected = 631.8965 - 0.009482565j
    print(E, np.isclose(E, expected))
    print()


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

    def compute_one_body(self, theta):
        return 2 * self._one_body_expression(theta)

    def compute_hartree(self, theta):
        return 2 * self._hartree_expression(theta, theta)

    def compute_exchange(self, theta):
        return -self._exchange_expression(theta, theta) - self._singularity_correction


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


def biorthogonalize(backend, trial, walkers):
    inverse_overlap = backend.linalg.inv(trial.T @ walkers)
    return contract("wpi, wij -> wpj", walkers, inverse_overlap)


if __name__ == "__main__":
    # main("single", "numpy")
    main("double", "numpy")
    # main("single", "jax")
    # main("double", "jax")
