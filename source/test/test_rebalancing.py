from vafpy.functions import rebalance_global 
from mpi4py import MPI 
from unittest.mock import patch
import numpy as np

def test_rebalance_reset_weight_to_one():
    comm = MPI.COMM_WORLD
    walkers_mat_up = np.zeros((100,1,1), dtype=np.complex128)
    initial_weights = np.random.random(100).astype(np.complex128)
    new_walker, weight = rebalance_global(comm, walkers_mat_up, initial_weights)
    assert np.all(weight == 1)

def test_rebalance():
    comm = MPI.COMM_WORLD
    walkers_mat_up = np.arange(10).astype(np.complex128).reshape((10, 1, 1))
    initial_weights = np.linspace(0.2, 2, 10).astype(np.complex128)
    with patch("numpy.random.random", return_value=0.3):
        new_walker, weight = rebalance_global(comm, walkers_mat_up, initial_weights)
    expected_walker = np.array([1, 3, 4, 5, 6, 7, 7, 8, 9, 9]).astype(np.complex128).reshape((10, 1, 1))
    assert np.allclose(new_walker, expected_walker)
    
    
