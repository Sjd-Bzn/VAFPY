from functions import propagate_walkers
from functions import Hamiltonian 
from functions import Configuration
from functions import Walkers
from functions import NumpyBackend
from functions import obtain_H2
from functions import obtain_H1
from functions import initialize_determinant 
import numpy as np
from mpi4py import MPI

def test_HF():
    max_seed = np.iinfo(np.int32).max
    seed = np.random.randint(max_seed)
    backend = NumpyBackend(seed)
    config = Configuration(
        num_walkers=200,
        num_kpoint=1,
        num_orbital=8,
        num_electron=4,
        num_g = 36,
        singularity=0,
        propagator="S2",
        order_propagation=6,
        timestep=0.00075,
        comm=MPI.COMM_WORLD,
        precision="Double",
        backend=backend,
        )
 
    hamiltonian = Hamiltonian(
        one_body=obtain_H1(config),
        two_body=obtain_H2(config),
        )
    
    trial_det, walkers = initialize_determinant(config)
    hamiltonian.setup_energy_expressions(config, trial_det)
    h_0 = 1.05
    e_0 = 0
    new_walkers, num_rare_event = propagate_walkers(config, trial_det, walkers, hamiltonian, h_0, e_0)
    



