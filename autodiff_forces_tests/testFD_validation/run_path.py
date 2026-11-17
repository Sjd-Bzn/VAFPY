"""Pathwise check: forward-mode derivative against the finite difference along the SAME trajectory (same random
numbers, same comb random number), windows of n_blocks x block steps from equilibrated walkers.
  off  : no population control -> the trajectory is smooth in lam: FD(h) -> AD(off) for h -> 0
  comb : population control after every block -> FD(h) -> AD(zero rule) for h -> 0, except in windows where the
         comb selects different walkers at +h and -h (a jump; probability ~ h)
usage: run_path.py CHAIN NW N_WINDOWS N_BLOCKS BLOCK h1,h2,... OUT.npz
"""
from common import *

chain, nw, nwin, nblk, block = (int(a) for a in sys.argv[1:6])
hs = [float(x) for x in sys.argv[6].split(",")]
out = sys.argv[7]
model, walkers = setup(nw, 4000 + chain)
key = jax.random.PRNGKey(400 + chain)
t0 = time.time()
d, w, e0 = equilibrate(model, walkers, nw, key, block)
print(f"chain {chain}: equilibrated E = {e0:.6f} ({time.time() - t0:.0f} s)", flush=True)
res = {k: [] for k in ("E", "ad_off", "ad_zero", "ad_martin", "fd_off", "fd_comb")}
for i in range(nwin):
    t1 = time.time()
    args = (d, w, e0, jax.random.fold_in(key, i), nblk, block)
    E, a_off = model.energy_and_derivative(0.0, *args, "off")
    E, a_zero = model.energy_and_derivative(0.0, *args, "zero")
    _, a_mart = model.energy_and_derivative(0.0, *args, "martin")
    f_off, f_comb = [], []
    for h in hs:
        f_off.append((model.window(h, *args, "off")[0] - model.window(-h, *args, "off")[0]) / (2 * h))
        f_comb.append((model.window(h, *args, "zero")[0] - model.window(-h, *args, "zero")[0]) / (2 * h))
    for k, v in zip(res, (E, a_off, a_zero, a_mart, np.array(f_off), np.array(f_comb))):
        res[k].append(np.asarray(v))
    _, d, w = model.window(0.0, *args, "zero")
    print(f"chain {chain} window {i}: ({time.time() - t1:.0f} s)", flush=True)
    np.savez(out, hs=np.array(hs), nw=nw, block=block, **{k: np.array(v) for k, v in res.items()})
print(f"chain {chain} done", flush=True)
