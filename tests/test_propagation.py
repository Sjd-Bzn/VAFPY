from unittest.mock import patch
import numpy as np
import numpy.testing as npt
from afqmc.propagation import Constants, State


def test_create_auxiliary_field_without_force_bias():
    num_g = 24
    num_band = 4
    num_walker = 3
    L = np.random.random((num_band, num_band, num_g))
    constants = Constants(L, num_walker)
    state = State(constants)
    x = np.random.normal(size=(num_g, num_walker))
    with patch.object(state, "random_normal", return_value=x) as mock:
        auxiliary_field = state.new_auxiliary_field(force_bias=None)
        mock.assert_called_once_with()
    npt.assert_allclose(auxiliary_field, L @ x)


def test_create_auxiliary_field_with_force_bias():
    num_g = 12
    num_band = 7
    num_walker = 4
    L = np.random.random((num_band, num_band, num_g))
    constants = Constants(L, num_walker)
    state = State(constants)
    x = np.random.normal(size=(num_g, num_walker))
    force_bias = np.random.random((num_g, num_walker))
    with patch.object(state, "random_normal", return_value=x) as mock:
        auxiliary_field = state.new_auxiliary_field(force_bias)
        mock.assert_called_once_with()
    npt.assert_allclose(auxiliary_field, L @ (x - force_bias))
