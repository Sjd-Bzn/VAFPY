"""Equilibration from the HF determinant followed by windows, on one trajectory.
Equilibration: after every block the energy E and the one- and two-body parts E1, E2 of normal AFQMC (trial on the
left, as measure_energy). Windows: after every block E, E1, E2 of normal AFQMC and the derivatives dE/dlam for
H1 -> (1 + lam) H1 (E1) and L -> (1 + lam/2) L (E2) with Martin's rule, the zero rule and without population control
(same random numbers). Every window starts from the end of the previous one (zero rule, lam = 0).
With a ninth argument "cap" the local energies of all estimators are capped to e0 +- sqrt(2 / dtau).
usage: run_eq_window.py CHAIN NW N_EQ_BLOCKS N_WINDOWS N_BLOCKS BLOCK OUT.npz [cap]"""
import os, sys, time
from functools import partial
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
ecap = "auto" if len(sys.argv) > 8 and sys.argv[8] == "cap" else None
config = make_config(nw, precision="Double", backend=new.JaxBackend(7000 + chain), nk=1, nb=16, ne=4, ng=136)
q = new.obtain_Q_list(config, os.path.join(P, "Q_list.npy")); h1 = new.obtain_H1(config, os.path.join(P, "H1_svd.npy"))
h2c, _ = new.obtain_H2_host(config, os.path.join(P, "H2_zip.npy")); sizes = new.obtain_Q_sizes(config, os.path.join(P, "Q_sizes.npy"))
H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
trial, walkers = new.initialize_determinant(config); H.setup_energy_expressions(config, trial)
L = np.asarray(h2c)
m1 = af.Model(config, H, trial, V=np.asarray(h1), ecap=ecap)
m2 = af.Model(config, H, trial, V=np.zeros_like(np.asarray(h1)), L=L, dL=0.5 * L, ecap=ecap)
d = jnp.asarray(walkers.slater_det, dtype=m1.ctype); w = jnp.ones(nw, dtype=m1.ctype)
e0 = float(m1.mixed_energy(0.0, d, w).real)


@partial(jax.jit, static_argnums=(4,))
def snapshots(d, w, e0, key, n):
    """n blocks of the trajectory of m1.window(0, ..., "zero"): after every block E, E1 and E2 of normal AFQMC."""
    P0 = m1.pieces(0.0)
    exp_half = af.expm_taylor(-0.5 * m1.dtau * P0["h1_total"])

    def one_block(carry, b):
        d, w = carry
        kb = jax.random.fold_in(key, b)

        def one_step(c, t):
            x = jax.random.normal(jax.random.fold_in(kb, t), (2 * m1.ng, m1.nw), dtype=m1.rtype)
            return m1.step(exp_half, e0, c[0], c[1], x, P0), None

        (d, w), _ = jax.lax.scan(one_step, (d, w), jnp.arange(block))
        e, v1 = jax.jvp(lambda l: m1.mixed_energy(l, d, w, e_ref=e0), (0.0,), (1.0,))
        _, v2 = jax.jvp(lambda l: m2.mixed_energy(l, d, w, e_ref=e0), (0.0,), (1.0,))
        u = jax.random.uniform(jax.random.fold_in(kb, block + 1), (), dtype=m1.rtype)
        d, w = m1.rebalance(d, w, u, "zero")
        return (d, w), (e, v1, v2)

    (d, w), r = jax.lax.scan(one_block, (d, w), jnp.arange(n))
    return r, d, w


t0 = time.time(); key = jax.random.PRNGKey(700 + chain)
eq = {k: [] for k in ("E", "E1", "E2")}
for c in range(neq // 10):                      # e0 updated every 10 blocks, as in run_window.py
    r, d, w = snapshots(d, w, e0, jax.random.fold_in(key, 10**6 + c), 10)
    for k, v in zip(eq, r):
        eq[k] += list(np.asarray(v).real)
    e0 = float(np.mean(eq["E"][len(eq["E"]) // 2:]))
print(f"chain {chain}: {neq * block} equilibration steps, E = {e0:.4f}  ({time.time() - t0:.0f} s)", flush=True)

names = ["E", "E1", "E2"] + [f"{q}_{m}" for m in ("martin", "zero", "off") for q in ("E1d", "E2d")] + ["E_off"]
res = {k: [] for k in names}
for i in range(nwin):
    k = jax.random.fold_in(key, i)
    args = (d, w, e0, k, nblk, block)
    r, _, _ = snapshots(d, w, e0, k, nblk)
    vals = list(r)
    for mode in ("martin", "zero", "off"):
        Em, g1 = m1.energy_and_derivative(0.0, *args, mode)
        _, g2 = m2.energy_and_derivative(0.0, *args, mode)
        vals += [g1, g2]
    vals.append(Em)                                # energy of the run without population control
    for n, v in zip(names, vals):
        res[n].append(np.asarray(v).real)
    _, d, w = m1.window(0.0, *args, "zero")
    if i % 10 == 9 or i == nwin - 1:
        np.savez(out, block=block, nw=nw, **{f"eq_{k}": np.array(v) for k, v in eq.items()},
                 **{n: np.array(v) for n, v in res.items()})
        print(f"chain {chain} window {i + 1}: E1 deriv (martin) at end {np.mean(np.array(res['E1d_martin'])[:, -1]):.3f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
print(f"chain {chain} done", flush=True)
