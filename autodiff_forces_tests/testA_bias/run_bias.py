"""Test A: bias of the rebalancing rule for dE/dlam (H1 -> H1 + lam V, V = H1); diamond primitive cell, Gamma only.

One independent Markov chain per process: equilibrate at lam = 0, then consecutive windows of n_blocks blocks of
`block` steps (population control after every block, as REBPRI = block in production). Every window starts from the
end of the previous window at lam = 0 (start state does not depend on lam). Per window, with the same random numbers:
  AD derivative (forward mode) with the rules martin, zero, off
  energies at lam = +h and -h with population control (comb) and without (off)
Mean over windows: AD(martin) and AD(zero) are compared with the finite difference of the mean energies, which is
the derivative of the expected energy (the comb jumps are included in the average).
usage: run_bias.py CHAIN NW N_WINDOWS N_BLOCKS BLOCK H1,H2,... OUT.npz
For a fixed trajectory and small h the comb picks the same walkers at lam +- h, so the finite difference of one
window equals the zero rule; the rebalancing contribution enters the mean through the rare windows where the comb
changes (large, rare jumps). Several h are used for the finite difference of the mean energies.
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
hs, out = [float(x) for x in sys.argv[6].split(",")], sys.argv[7]

config = make_config(nw, precision="Double", backend=new.JaxBackend(1000 + chain), nk=1, nb=16, ne=4, ng=136)
q = new.obtain_Q_list(config, os.path.join(P, "Q_list.npy"))
h1 = new.obtain_H1(config, os.path.join(P, "H1_svd.npy"))
h2c, _ = new.obtain_H2_host(config, os.path.join(P, "H2_zip.npy"))
sizes = new.obtain_Q_sizes(config, os.path.join(P, "Q_sizes.npy"))
H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
trial, walkers = new.initialize_determinant(config)
H.setup_energy_expressions(config, trial)
model = af.Model(config, H, trial, V=np.asarray(h1))
key = jax.random.PRNGKey(chain)

d = jnp.asarray(walkers.slater_det, dtype=model.ctype)
w = jnp.ones(nw, dtype=model.ctype)
e0 = float(model.mixed_energy(0.0, d, w))
# equilibration: 5 x 10 blocks of `block` steps, e0 updated to the running mean of the second half
t0 = time.time()
eq = []
for b in range(5):
    en, d, w = model.window(0.0, d, w, e0, jax.random.fold_in(key, 10**6 + b), 10, block, "zero")
    eq += list(np.asarray(en)); e0 = float(np.mean(eq[len(eq) // 2:]))
print(f"chain {chain}: equilibrated E = {e0:.6f}  ({time.time() - t0:.1f} s)", flush=True)

res = {k: [] for k in ("E", "dE_martin", "dE_zero", "dE_off", "E_off", "Ep", "Em", "Ep_off", "Em_off")}
for i in range(nwin):
    t1 = time.time()
    k = jax.random.fold_in(key, i)
    args = (d, w, e0, k, nblk, block)
    E, dm = model.energy_and_derivative(0.0, *args, "martin")
    _, dz = model.energy_and_derivative(0.0, *args, "zero")
    Eo, do = model.energy_and_derivative(0.0, *args, "off")
    Ep = np.array([model.window(h, *args, "martin")[0] for h in hs]); Em = np.array([model.window(-h, *args, "martin")[0] for h in hs])
    Epo = model.window(hs[0], *args, "off")[0]; Emo = model.window(-hs[0], *args, "off")[0]
    for name, v in zip(res, (E, dm, dz, do, Eo, Ep, Em, Epo, Emo)):
        res[name].append(np.asarray(v))
    _, d, w = model.window(0.0, *args, "zero")            # next window starts here (lam = 0)
    if i < 3 or i % 20 == 0:
        print(f"chain {chain} window {i}: E {float(E[-1]):.5f} dE martin {float(dm[-1]):.4f} zero {float(dz[-1]):.4f}"
              f" off {float(do[-1]):.4f} FD " + " ".join(f"{float((Ep[j,-1]-Em[j,-1])/(2*h)):.4f}" for j, h in enumerate(hs)) + f"  ({time.time() - t1:.1f} s)", flush=True)
    if i % 10 == 9 or i == nwin - 1:
        np.savez(out, h=np.array(hs), nw=nw, block=block, n_blocks=nblk, e0=e0, **{k: np.array(v) for k, v in res.items()})
print(f"chain {chain} done", flush=True)
