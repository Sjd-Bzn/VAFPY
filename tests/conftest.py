import random

import numpy as np
import pytest

from afqmc.constants import Constants


@pytest.fixture
def make_constants():
    def inner(number_orbital=None, number_g=None, **user_options):
        number_orbital = number_orbital or random.randrange(4, 15)
        number_g = number_g or random.randrange(4, 30)
        default_options = {
            "number_electron": random.randrange(1, number_orbital),
            "number_walker": random.randrange(1, 20),
            "number_k": 1,
            "tau": random.uniform(1e-4, 1e-3),
        }
        options = {**default_options, **user_options}
        L = np.random.random((number_orbital, number_orbital, number_g))
        return Constants(L, **options)

    return inner
