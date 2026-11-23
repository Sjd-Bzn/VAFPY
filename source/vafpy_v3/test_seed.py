"""Seed of a run and the random-number streams of the MPI ranks (functions.run_seed, functions.seed_sequence)."""
import numpy as np
import pytest
from mpi4py import MPI

import functions as new

BACKENDS = ["numpy"]
try:
    import jax  # noqa: F401
    BACKENDS.append("jax")
except ImportError:
    pass


def test_seed_from_input():
    assert new.run_seed(41, MPI.COMM_WORLD) == (41, "input")
    assert new.run_seed("41", MPI.COMM_WORLD) == (41, "input")


@pytest.mark.parametrize("value", [None, "random", "Random", " RANDOM "])
def test_random_seed_is_drawn_and_reported(value):
    s1, src1 = new.run_seed(value, MPI.COMM_WORLD)
    s2, src2 = new.run_seed(value, MPI.COMM_WORLD)
    assert src1 == src2 == "random"
    assert 0 <= s1 < 2**63 and 0 <= s2 < 2**63
    assert s1 != s2


def test_rank_stream_is_the_spawned_child():
    children = np.random.SeedSequence(41).spawn(8)
    for r, child in enumerate(children):
        a = new.seed_sequence(41, r).generate_state(4)
        assert np.array_equal(a, child.generate_state(4))


@pytest.mark.parametrize("name", BACKENDS)
def test_same_seed_and_rank_reproduce_the_numbers(name):
    draw = lambda: np.asarray(new.make_backend(name, new.seed_sequence(123456789012345, 3)).random_normal((50, 7), np.float64))
    assert np.array_equal(draw(), draw())


@pytest.mark.parametrize("name", BACKENDS)
def test_ranks_and_seeds_give_different_independent_streams(name):
    n = 20000
    x = {(s, r): np.asarray(new.make_backend(name, new.seed_sequence(s, r)).random_normal((n,), np.float64))
         for s in (41, 42) for r in (0, 1, 2)}
    keys = list(x)
    for i, a in enumerate(keys):
        assert abs(x[a].mean()) < 5 / np.sqrt(n) and abs(x[a].std() - 1) < 0.05
        for b in keys[i + 1:]:
            assert not np.array_equal(x[a], x[b])
            assert abs(np.corrcoef(x[a], x[b])[0, 1]) < 5 / np.sqrt(n)


@pytest.mark.parametrize("name", BACKENDS)
def test_integer_seed_still_accepted(name):
    a = np.asarray(new.make_backend(name, 5).random_uniform((10,)))
    b = np.asarray(new.make_backend(name, new.seed_sequence(5)).random_uniform((10,)))
    assert np.array_equal(a, b)


def test_random_seed_is_the_same_on_all_ranks():
    comm = MPI.COMM_WORLD
    seed, _ = new.run_seed("random", comm)
    assert all(s == seed for s in comm.allgather(seed))
