"""Common setup for the finite-difference validation (diamond primitive cell at Gamma, V = H1)."""
import os, sys, time
import numpy as np
SRC = "/fs/home/sajad/projects/git/AFQMC-autodiff-forces/source/vafpy_v3"
sys.path.insert(0, SRC)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functions as new
import autodiff_forces as af
from test_compact_hamiltonian import make_config
from test_autodiff_forces import GAMMA as P


def setup(nw, seed):
    config = make_config(nw, precision="Double", backend=new.JaxBackend(seed), nk=1, nb=16, ne=4, ng=136)
    q = new.obtain_Q_list(config, os.path.join(P, "Q_list.npy"))
    h1 = new.obtain_H1(config, os.path.join(P, "H1_svd.npy"))
    h2c, _ = new.obtain_H2_host(config, os.path.join(P, "H2_zip.npy"))
    sizes = new.obtain_Q_sizes(config, os.path.join(P, "Q_sizes.npy"))
    H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    return af.Model(config, H, trial, V=np.asarray(h1)), walkers


def equilibrate(model, walkers, nw, key, block, n=20):
    """n x 10 blocks at lam = 0 from the trial; returns walkers, weights and the energy shift e0."""
    d = jnp.asarray(walkers.slater_det, dtype=model.ctype)
    w = jnp.ones(nw, dtype=model.ctype)
    e0 = float(model.mixed_energy(0.0, d, w))
    eq = []
    for b in range(n):
        en, d, w = model.window(0.0, d, w, e0, jax.random.fold_in(key, 10**6 + b), 10, block, "zero")
        eq += list(np.asarray(en)); e0 = float(np.mean(eq[len(eq) // 2:]))
    return d, w, e0
