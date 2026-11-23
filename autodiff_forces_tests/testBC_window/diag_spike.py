"""Which walkers make a spike in the derivative? Replays chain CHAIN of run_eq_window.py up to window WIN, then
splits the derivative of the last block energy into walker contributions
    dE = sum_w [ dW_w (E_w - E) + W_w dE_w ] / sum W
usage: diag_spike.py CHAIN WIN MODE"""
import os, sys
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

chain, win, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
nw, neq, nblk, block = 100, 300, 40, 10
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
for c in range(neq // 10):
    E, d, w = m1.window(0.0, d, w, e0, jax.random.fold_in(key, 10**6 + c), 10, block, "zero")
    eqE += list(np.asarray(E).real); e0 = float(np.mean(eqE[len(eqE) // 2:]))
for i in range(win):
    _, d, w = m1.window(0.0, d, w, e0, jax.random.fold_in(key, i), nblk, block, "zero")
k = jax.random.fold_in(key, win)

for name, m in (("E1", m1), ("E2", m2)):
    def g(l):
        _, dd, ww = m.window(l, d, w, e0, k, nblk - 1, block, mode)
        Pl = m.pieces(l); exp_half = af.expm_taylor(-0.5 * m.dtau * Pl["h1_total"])
        kb = jax.random.fold_in(k, nblk - 1)
        for t in range(block):
            x = jax.random.normal(jax.random.fold_in(kb, t), (2 * m.ng, m.nw), dtype=m.rtype)
            dd, ww = m.step(exp_half, e0, dd, ww, x, Pl)
        return m.local_energy(l, dd, Pl), ww, m.overlap(dd)
    (Ew, W, ov), (dEw, dW, _) = jax.jvp(g, (0.0,), (1.0,))
    Ew, W, dEw, dW = (np.asarray(a) for a in (Ew, W, dEw, dW)); ov = np.abs(np.asarray(ov))
    S = W.real.sum(); E = (W * Ew).real.sum() / S
    cw = (dW * (Ew - E) + W * dEw).real / S
    print(f"\n{name} chain {chain} window {win} {mode}: E = {E:.3f}, dE = {cw.sum():.3f}, alive {np.sum(W.real > 0)}")
    print("  walker  contrib   W      |ovl|/median  E_w - E      dE_w     dW/W")
    for j in np.argsort(-np.abs(cw))[:6]:
        print(f"  {j:4d} {cw[j]:9.3f} {W[j].real:6.3f} {ov[j] / np.median(ov):10.2e} {(Ew[j] - E).real:9.2f}"
              f" {dEw[j].real:10.2f} {(dW[j] / W[j]).real if W[j].real else 0:9.2f}")
    print(f"  median |cw| {np.median(np.abs(cw)):.3f}; smallest |ovl|/median {np.min(ov[W.real > 0]) / np.median(ov):.1e}")
