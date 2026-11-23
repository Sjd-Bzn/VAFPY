"""Along every window: E1 and E2 measured on the walkers after each block (normal AFQMC, mixed formula) and E1, E2 as
derivatives (Martin's rule), on the same trajectory (same keys, fields and comb as Model.window at lam = 0).
usage: e1e2_window.py CHAIN NW N_WINDOWS N_BLOCKS BLOCK OUT.npz"""
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

chain, nw, nwin, nblk, block = (int(a) for a in sys.argv[1:6]); out = sys.argv[6]
config = make_config(nw, precision="Double", backend=new.JaxBackend(5000 + chain), nk=1, nb=16, ne=4, ng=136)
q = new.obtain_Q_list(config, os.path.join(P, "Q_list.npy")); h1 = new.obtain_H1(config, os.path.join(P, "H1_svd.npy"))
h2c, _ = new.obtain_H2_host(config, os.path.join(P, "H2_zip.npy")); sizes = new.obtain_Q_sizes(config, os.path.join(P, "Q_sizes.npy"))
H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
trial, walkers = new.initialize_determinant(config); H.setup_energy_expressions(config, trial)
L = np.asarray(h2c)
m1 = af.Model(config, H, trial, V=np.asarray(h1))
m2 = af.Model(config, H, trial, V=np.zeros_like(np.asarray(h1)), L=L, dL=0.5 * L)
d = jnp.asarray(walkers.slater_det, dtype=m1.ctype); w = jnp.ones(nw, dtype=m1.ctype)
e0 = float(m1.mixed_energy(0.0, d, w).real)


@jax.jit
def snapshots(d, w, e0, key):
    """The trajectory of m1.window(0, ...): after every block E, E1 and E2 measured on the walkers."""
    P0 = m1.pieces(0.0)
    exp_half = af.expm_taylor(-0.5 * m1.dtau * P0["h1_total"])

    def one_block(carry, b):
        d, w = carry
        kb = jax.random.fold_in(key, b)

        def one_step(c, t):
            x = jax.random.normal(jax.random.fold_in(kb, t), (2 * m1.ng, m1.nw), dtype=m1.rtype)
            return m1.step(exp_half, e0, c[0], c[1], x, P0), None

        (d, w), _ = jax.lax.scan(one_step, (d, w), jnp.arange(block))
        e, v1 = jax.jvp(lambda l: m1.mixed_energy(l, d, w), (0.0,), (1.0,))
        _, v2 = jax.jvp(lambda l: m2.mixed_energy(l, d, w), (0.0,), (1.0,))
        u = jax.random.uniform(jax.random.fold_in(kb, block + 1), (), dtype=m1.rtype)
        d, w = m1.rebalance(d, w, u, "zero")
        return (d, w), (e, v1, v2)

    _, r = jax.lax.scan(one_block, (d, w), jnp.arange(nblk))
    return r


key = jax.random.PRNGKey(900 + chain); eq = []; t0 = time.time()
for b in range(20):
    en, d, w = m1.window(0.0, d, w, e0, jax.random.fold_in(key, 10**6 + b), 10, block, "zero")
    eq += list(np.asarray(en)); e0 = float(np.mean(eq[len(eq) // 2:]))
print(f"chain {chain}: equilibrated E = {e0:.4f} ({time.time() - t0:.0f} s)", flush=True)
res = {k: [] for k in ("E", "E1_walkers", "E2_walkers", "E_check", "E1_deriv", "E2_deriv")}
for i in range(nwin):
    k = jax.random.fold_in(key, i)
    args = (d, w, e0, k, nblk, block)
    es, v1, v2 = snapshots(d, w, e0, k)
    E, g1 = m1.energy_and_derivative(0.0, *args, "martin")
    _, g2 = m2.energy_and_derivative(0.0, *args, "martin")
    for name, v in zip(res, (es, v1, v2, E, g1, g2)):
        res[name].append(np.asarray(v).real)
    _, d, w = m1.window(0.0, *args, "zero")
    if i % 10 == 9 or i == nwin - 1:
        np.savez(out, block=block, **{n: np.array(v) for n, v in res.items()})
        print(f"chain {chain} window {i + 1}: max |E - E_check| {np.abs(np.array(res['E']) - np.array(res['E_check'])).max():.2e}"
              f"  ({time.time() - t0:.0f} s)", flush=True)
