"""Setup for the AFQMC force at r003 (diamond primitive cell, Gamma, atom 2 at +0.03 A along x): H1, L at r003 and
the derivatives dH1/dx, dL/dx (per A, central difference +-0.01 A, orbitals of r003), jointly compressed
(../svd_joint, threshold 1e-5, 173 columns). lam is the displacement of atom 2 along x in A."""
import os, sys, time
import numpy as np
SRC = os.environ.get("VAFPY_SRC", "/fs/home/sajad/projects/git/AFQMC-autodiff-forces/source/vafpy_v3")
sys.path.insert(0, SRC)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functions as new
import autodiff_forces as af
from test_compact_hamiltonian import make_config
S = os.environ.get("FORCE_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "svd_joint"))
Q = os.path.join(os.environ.get("AFQMC_DATA", "/fs/home/sajad/projects/data"), "test", "diamond_gamma_prim", "Q_list.npy")
TAG = "1e-05"
E_HF, DE_HF = 113.6480608517, 0.86412401          # electronic HF energy and dE/dx (walkers = trial)
DX_CLASSICAL = 2.546980 + 0.004807                # d(TEWEN + PAWdc)/dx, eV/A


def setup(nw, seed):
    Lc, dL = np.load(f"{S}/H2_zip_{TAG}.npy"), np.load(f"{S}/dH2_zip_{TAG}.npy")
    config = make_config(nw, precision="Double", backend=new.JaxBackend(seed), nk=1, nb=16, ne=4, ng=Lc.shape[2])
    q = new.obtain_Q_list(config, Q)
    H = new.build_hamiltonian(config, new.obtain_H1(config, f"{S}/H1_c0.npy"), Lc, "compact", q)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    dh1 = np.asarray(new.obtain_H1(config, f"{S}/dH1.npy"))
    return af.Model(config, H, trial, V=dh1, L=Lc, dL=dL), walkers


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
