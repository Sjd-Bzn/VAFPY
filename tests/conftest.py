import random

import numpy as np
import pytest

from afqmc.constants import Constants


@pytest.fixture(autouse=True, scope="session")
def seed_random_numbers():
    random.seed(811972495)
    np.random.seed(526809193)


@pytest.fixture
def make_constants():
    def inner(number_orbital=None, number_g=None, number_k=1, **user_options):
        number_orbital = number_orbital or random.randrange(4, 15)
        number_g = number_g or random.randrange(4, 30)
        default_options = {
            "number_electron": random.randrange(1, number_orbital),
            "number_walker": random.randrange(1, 20),
            "tau": random.uniform(1e-4, 1e-3),
        }
        options = {**default_options, **user_options}
        H1 = np.random.random((number_k, number_orbital, number_orbital))
        H1T = np.moveaxis(H1, 1, -1)
        H1 = 0.5 * (H1 + H1T)  # make Hermitian
        shape = number_k * np.array((number_orbital, number_orbital, number_g))
        L = np.random.random(shape) + 1j * np.random.random(shape)
        return Constants(H1, L, **options)

    return inner
