"""Mixed estimator of <H1> (V = H1), for comparison with the derivative against the window length (tests B and C).

Same system, walkers, time step, population control and equilibration as run_window.py. After equilibration the chain
is propagated at lam = 0; after every block the mixed energy E and the mixed estimator of V,
<V>_mixed = sum_w W_w <Psi_T|V|Phi_w>/<Psi_T|Phi_w> / sum_w W_w, are recorded. <V>_mixed is obtained as the derivative
of the mixed energy with respect to lam at fixed walkers (forward mode), i.e. the explicit part of dE/dlam only, which
is the value of the window derivative directly after its reset.
usage: run_mixed.py CHAIN NW N_SEGMENTS N_BLOCKS BLOCK OUT.npz
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

chain, nw, nseg, nblk, block = (int(a) for a in sys.argv[1:6])
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
v_hf = float(jax.jvp(lambda l: model.mixed_energy(l, d, w), (0.0,), (1.0,))[1])     # walkers = trial: HF value
t0 = time.time()
eq = []
for b in range(20):                      # equilibration as in run_window.py: 2000 steps for block 10
    en, d, w = model.window(0.0, d, w, e0, jax.random.fold_in(key, 10**6 + b), 10, block, "zero")
    eq += list(np.asarray(en)); e0 = float(np.mean(eq[len(eq) // 2:]))
print(f"chain {chain}: HF <V> = {v_hf:.6f}; equilibrated E = {e0:.6f}  ({time.time() - t0:.1f} s)", flush=True)


@jax.jit
def segment(d, w, k):
    """nblk blocks of `block` steps at lam = 0; after every block E and <V>_mixed (before the population control)."""
    P0 = model.pieces(0.0)
    exp_half = af.expm_taylor(-0.5 * model.dtau * P0["h1_total"])

    def one_block(carry, b):
        d, w = carry
        kb = jax.random.fold_in(k, b)

        def one_step(c, t):
            dd, ww = c
            x = jax.random.normal(jax.random.fold_in(kb, t), (2 * model.ng, model.nw), dtype=model.rtype)
            return model.step(exp_half, e0, dd, ww, x, P0), None

        (d, w), _ = jax.lax.scan(one_step, (d, w), jnp.arange(block))
        e, v = jax.jvp(lambda l: model.mixed_energy(l, d, w), (0.0,), (1.0,))
        u = jax.random.uniform(jax.random.fold_in(kb, block + 1), (), dtype=model.rtype)
        d, w = model.rebalance(d, w, u, "zero")
        return (d, w), (e, v)

    (d, w), (es, vs) = jax.lax.scan(one_block, (d, w), jnp.arange(nblk))
    return d, w, es, vs


E, V = [], []
for i in range(nseg):
    d, w, es, vs = segment(d, w, jax.random.fold_in(key, i))
    E.append(np.asarray(es)); V.append(np.asarray(vs))
    if i % 10 == 0 or i == nseg - 1:
        print(f"chain {chain} segment {i}: E {np.mean(E):.4f}  <V>_mixed {np.mean(V):.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
        np.savez(out, nw=nw, block=block, v_hf=v_hf, e0=e0, E=np.concatenate(E), V=np.concatenate(V))
np.savez(out, nw=nw, block=block, v_hf=v_hf, e0=e0, E=np.concatenate(E), V=np.concatenate(V))
print(f"chain {chain} done", flush=True)
