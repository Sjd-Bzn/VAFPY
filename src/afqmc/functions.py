from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from opt_einsum import contract_expression, contract


@dataclass
class Configuration:
    num_walkers: int  # number of walkers per core
    num_kpoint: int  # number of k-points to sample the Brillouin zone
    num_orbital: int  # size of the basis
    num_electron: int  # number of occupied states
    singularity: float  # singularity used for the G=0 component (fsg)
    comm: MPI.Comm  # communicator over which the walkers are distributed


@dataclass
class Hamiltonian:
    one_body: np.typing.ArrayLike
    # one-body part of Hamiltonian, expected shape (num_orbital, num_orbital, num_kpoint)
    two_body: np.typing.ArrayLike
    # two-body part of Hamiltonian after factorization,
    # expected shape (num_orbital, num_orbital * num_kpoint, num_g * num_kpoint)

    def setup_energy_expressions(self, config, trial_state):
        alp = np.einsum('pi, prg -> irg', trial_state, self.two_body)
        alp_t = np.einsum('pi, rpg -> irg', trial_state, self.two_body.conj())
        alp_s = alp.astype(np.complex64)
        alp_s_t = alp_t.astype(np.complex64)
        self.compute_exchange = contract_expression('Nri,jrG,Npj,ipG->N',(config.num_walkers,config.num_orbital*config.num_kpoint,config.num_electron*config.num_kpoint),alp_s,(config.num_walkers,config.num_orbital*config.num_kpoint,config.num_electron*config.num_kpoint),alp_s_t,constants=[1,3],optimize='greedy')
        self.compute_hartree = contract_expression('Nri,irG,Npj,jpG->N',(config.num_walkers,config.num_orbital*config.num_kpoint,config.num_electron*config.num_kpoint),alp_s,(config.num_walkers,config.num_orbital*config.num_kpoint,config.num_electron*config.num_kpoint),alp_s_t,constants=[1,3],optimize='greedy')

def main():
    config = Configuration(
        num_walkers= 10,
        num_kpoint=1,
        num_orbital=8,
        num_electron=4,
        singularity=0,
        comm=MPI.COMM_WORLD,
    )
    hamiltonian = Hamiltonian(
        one_body = np.load("H1.npy"),
        two_body = np.moveaxis(np.load("H2.npy"), 0, -1),
    )
    trial_state = np.eye(config.num_orbital,config.num_electron)
    hamiltonian.setup_energy_expressions(config, trial_state)
    mats_up = np.array(config.num_walkers * [trial_state], dtype=np.complex64)
    weights = np.ones(config.num_walkers, dtype=np.complex64)

    E = measure_E_gs_single(config,trial_state,weights,mats_up,hamiltonian)
    expected = 631.88947-2.8974954e-09j
    print(E, np.isclose(E, expected))
    np.random.seed(1887431)
    change = 0.05 * np.random.rand(*mats_up.shape)
    E = measure_E_gs_single(config,trial_state,weights,mats_up + change,hamiltonian)
    expected = 631.8965-0.009482565j
    print(E, np.isclose(E, expected))

def measure_E_gs_single(config,trial,weights,walkers,hamiltonian):
    thetas=[]
    for i in range(config.num_walkers):
        thetas.append(theta(trial, walkers[i]))
    thetas=np.array(thetas).astype(np.complex64)
    b=np.squeeze(np.dot(trial.T,hamiltonian.one_body)).astype(np.complex64)
    e1=2*contract('iNi->N',np.tensordot(b, thetas,axes=((1,1)))) / config.num_kpoint
    har_list  = 2* hamiltonian.compute_hartree(thetas,thetas) / config.num_kpoint
    exch_list = (-hamiltonian.compute_exchange(thetas,thetas) - config.num_electron * config.num_kpoint * config.singularity)/config.num_kpoint
    print(e1[0], har_list[0], exch_list[0])
    e_locs=e1+har_list+exch_list
    weights = weights.astype(np.complex64)
    val=0
    for e_loc, weight in zip(e_locs, weights):
        val+=e_loc*weight
    val_glb= config.comm.allreduce(val)/config.comm.Get_size()
    sum_w = np.sum(weights)
    sum_glb= config.comm.allreduce(sum_w)/config.comm.Get_size()
    return val_glb/sum_glb

def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial,walker)))

def overlap(left_slater_det, right_slater_det):
    '''
    It computes the overlap between two Slater determinants.
    '''
    overlap_mat = np.dot(left_slater_det.transpose(),right_slater_det)
    return overlap_mat

if __name__ == "__main__":
    main()
