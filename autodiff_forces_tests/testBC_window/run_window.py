"""Tests B and C: dE/dlam against the window length (H1 -> H1 + lam V, V = H1); diamond primitive cell, Gamma only.

One independent Markov chain per process: equilibrate at lam = 0, then consecutive windows of n_blocks blocks of
`block` steps (population control after every block, as REBPRI = block in production). The derivative after L steps
of a long window equals the derivative of a window of length L (the propagation is causal), so one window of
n_blocks*block steps gives dE/dlam for all lengths L = block, 2 block, ... . Per window: forward-mode derivative with
Martin's rule, the zero rule and without population control (same random numbers). Every window starts from the end
of the previous one at lam = 0.
usage: run_window.py CHAIN NW N_WINDOWS N_BLOCKS BLOCK OUT.npz
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

chain, nw, nwin, nblk, block = (int(a) for a in sys.argv[1:6])
out = sys.argv[6]
config = make_config(nw, precision="Double", backend=new.JaxBackend(2000 + chain), nk=1, nb=16, ne=4, ng=136)
q = new.obtain_Q_list(config, os.path.join(P, "Q_list.npy"))
h1 = new.obtain_H1(config, os.path.join(P, "H1_svd.npy"))
h2c, _ = new.obtain_H2_host(config, os.path.join(P, "H2_zip.npy"))
sizes = new.obtain_Q_sizes(config, os.path.join(P, "Q_sizes.npy"))
H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
trial, walkers = new.initialize_determinant(config)
H.setup_energy_expressions(config, trial)
model = af.Model(config, H, trial, V=np.asarray(h1))
key = jax.random.PRNGKey(100 + chain)

d = jnp.asarray(walkers.slater_det, dtype=model.ctype)
w = jnp.ones(nw, dtype=model.ctype)
e0 = float(model.mixed_energy(0.0, d, w))
t0 = time.time()
eq = []
for b in range(20):                      # equilibration: 20 x 10 blocks of `block` steps (2000 steps for block 10)
    en, d, w = model.window(0.0, d, w, e0, jax.random.fold_in(key, 10**6 + b), 10, block, "zero")
    eq += list(np.asarray(en)); e0 = float(np.mean(eq[len(eq) // 2:]))
print(f"chain {chain}: equilibrated E = {e0:.6f}  ({time.time() - t0:.1f} s)", flush=True)

res = {k: [] for k in ("E", "dE_martin", "dE_zero", "dE_off", "E_off")}
for i in range(nwin):
    t1 = time.time()
    args = (d, w, e0, jax.random.fold_in(key, i), nblk, block)
    E, dm = model.energy_and_derivative(0.0, *args, "martin")
    _, dz = model.energy_and_derivative(0.0, *args, "zero")
    Eo, do = model.energy_and_derivative(0.0, *args, "off")
    for name, v in zip(res, (E, dm, dz, do, Eo)):
        res[name].append(np.asarray(v))
    _, d, w = model.window(0.0, *args, "zero")
    if i < 3 or i % 20 == 0:
        print(f"chain {chain} window {i}: dE after {block}/{nblk//2*block}/{nblk*block} steps: martin "
              f"{float(dm[0]):.3f} {float(dm[nblk//2-1]):.3f} {float(dm[-1]):.3f}  zero {float(dz[-1]):.3f}"
              f"  off {float(do[-1]):.3f}  ({time.time() - t1:.1f} s)", flush=True)
    if i % 10 == 9 or i == nwin - 1:
        np.savez(out, nw=nw, block=block, n_blocks=nblk, e0=e0, **{k: np.array(v) for k, v in res.items()})
print(f"chain {chain} done", flush=True)
