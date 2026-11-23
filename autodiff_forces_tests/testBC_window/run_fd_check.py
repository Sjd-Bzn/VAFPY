"""Central finite differences with tiny steps and the same random numbers against the forward-mode derivative.
For one trajectory (same key) E_b(lam) is a smooth function of lam except at rare kinks (phaseless cut, caps) and,
with population control, jumps of the comb. So (E_b(h) - E_b(-h)) / 2h must equal the AD derivative of the same
path: exactly for "off" (no population control) and for "zero" (pathwise derivative with the comb indices fixed)
unless a comb decision flips between -h and +h. Martin's rule is not a pathwise derivative and is not checked.
Same chains, seeds, equilibration and windows as run_eq_window.py.
usage: run_fd_check.py CHAIN NW N_EQ_BLOCKS N_WINDOWS N_BLOCKS BLOCK OUT.npz"""
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

chain, nw, neq, nwin, nblk, block = (int(a) for a in sys.argv[1:7]); out = sys.argv[7]
H_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
config = make_config(nw, precision="Double", backend=new.JaxBackend(7000 + chain), nk=1, nb=16, ne=4, ng=136)
q = new.obtain_Q_list(config, os.path.join(P, "Q_list.npy")); h1 = new.obtain_H1(config, os.path.join(P, "H1_svd.npy"))
h2c, _ = new.obtain_H2_host(config, os.path.join(P, "H2_zip.npy")); sizes = new.obtain_Q_sizes(config, os.path.join(P, "Q_sizes.npy"))
H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
trial, walkers = new.initialize_determinant(config); H.setup_energy_expressions(config, trial)
L = np.asarray(h2c)
m1 = af.Model(config, H, trial, V=np.asarray(h1))
m2 = af.Model(config, H, trial, V=np.zeros_like(np.asarray(h1)), L=L, dL=0.5 * L)
d = jnp.asarray(walkers.slater_det, dtype=m1.ctype); w = jnp.ones(nw, dtype=m1.ctype)
e0 = float(m1.mixed_energy(0.0, d, w).real)

key = jax.random.PRNGKey(700 + chain)
eqE = []
for c in range(neq // 10):                                   # equilibration exactly as run_eq_window.py
    E, d, w = m1.window(0.0, d, w, e0, jax.random.fold_in(key, 10**6 + c), 10, block, "zero")
    eqE += list(np.asarray(E).real)
    e0 = float(np.mean(eqE[len(eqE) // 2:]))
print(f"chain {chain}: {neq * block} equilibration steps, E = {e0:.4f}", flush=True)

res = {f"{p}_{m}_{k}": [] for p in ("E1", "E2") for m in ("off", "zero") for k in ["ad"] + [f"fd{h:g}" for h in H_STEPS]}
t0 = time.time()
for i in range(nwin):
    k = jax.random.fold_in(key, i)
    args = (d, w, e0, k, nblk, block)
    for p, m in (("E1", m1), ("E2", m2)):
        for mode in ("off", "zero"):
            _, g = m.energy_and_derivative(0.0, *args, mode)
            res[f"{p}_{mode}_ad"].append(np.asarray(g).real)
            for h in H_STEPS:
                ep = np.asarray(m.window(h, *args, mode)[0]).real
                em = np.asarray(m.window(-h, *args, mode)[0]).real
                res[f"{p}_{mode}_fd{h:g}"].append((ep - em) / (2 * h))
    _, d, w = m1.window(0.0, *args, "zero")
    msg = "  ".join(f"{p} {mode} h=1e-5: {np.max(np.abs(res[f'{p}_{mode}_fd1e-05'][-1] - res[f'{p}_{mode}_ad'][-1])):.1e}"
                    for p in ("E1", "E2") for mode in ("off", "zero"))
    print(f"chain {chain} window {i + 1}: max|fd - ad| {msg}  ({time.time() - t0:.0f} s)", flush=True)
    np.savez(out, block=block, nw=nw, h=np.array(H_STEPS), **{n: np.array(v) for n, v in res.items()})
print(f"chain {chain} done", flush=True)
