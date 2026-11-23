"""Test C reference: AFQMC energy at fixed lam (H1 -> H1 + lam V, V = H1), one long run per process, no derivative.
d<E>/dlam of the converged projection = (E(+lam) - E(-lam)) / 2 lam from independent runs.
usage: run_lambda.py SEED LAM NW N_BLOCKS BLOCK OUT.npz [PART]
PART = h1 (default): H1 -> (1 + lam) H1;  PART = h2: L -> (1 + lam/2) L, i.e. H2 scaled (slope at lam = 0 is E2).
"""
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

seed, lam, nw, nblk, block, out = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
part = sys.argv[7] if len(sys.argv) > 7 else "h1"
config = make_config(nw, precision="Double", backend=new.JaxBackend(seed), nk=1, nb=16, ne=4, ng=136)
q = new.obtain_Q_list(config, os.path.join(P, "Q_list.npy"))
h1 = new.obtain_H1(config, os.path.join(P, "H1_svd.npy"))
h2c, _ = new.obtain_H2_host(config, os.path.join(P, "H2_zip.npy"))
sizes = new.obtain_Q_sizes(config, os.path.join(P, "Q_sizes.npy"))
H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
trial, walkers = new.initialize_determinant(config)
H.setup_energy_expressions(config, trial)
if part == "h2":
    L = np.asarray(h2c)
    model = af.Model(config, H, trial, V=np.zeros_like(np.asarray(h1)), L=L, dL=0.5 * L)
else:
    model = af.Model(config, H, trial, V=np.asarray(h1))
key = jax.random.PRNGKey(seed)
d = jnp.asarray(walkers.slater_det, dtype=model.ctype)
w = jnp.ones(nw, dtype=model.ctype)
e0 = float(model.mixed_energy(lam, d, w))
E = []
t0 = time.time()
for c in range(nblk // 100):
    en, d, w = model.window(lam, d, w, e0, jax.random.fold_in(key, c), 100, block, "zero")
    E += list(np.asarray(en))
    e0 = float(np.mean(E[len(E) // 2:]))
    np.savez(out, lam=lam, nw=nw, block=block, E=np.array(E))
    print(f"seed {seed} lam {lam}: {len(E)} blocks, mean E (second half) {e0:.5f}  ({time.time() - t0:.0f} s)", flush=True)
