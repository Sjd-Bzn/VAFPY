"""AFQMC force (dE/dx, lam = displacement in A): derivative of the expected energy after L steps, including population control, from a correlated
finite difference. Every window starts from the walkers of the lam = 0 chain; the windows at +h and -h use the same
random numbers (fields and comb random number), but the comb acts on the walkers of each lam, i.e. it is sampled
correctly at each lam. The mean over windows of (E_L(+h) - E_L(-h)) / 2h estimates d<E_L>/dlam + O(h^2).
Per window also: forward-mode derivative with Martin's rule and with the zero rule (same random numbers).
usage: run_fd.py CHAIN NW N_WINDOWS N_BLOCKS BLOCK h1,h2,... OUT.npz
"""
from common import *

chain, nw, nwin, nblk, block = (int(a) for a in sys.argv[1:6])
hs = [float(x) for x in sys.argv[6].split(",")]
out = sys.argv[7]
model, walkers = setup(nw, 3000 + chain)
key = jax.random.PRNGKey(300 + chain)
t0 = time.time()
d, w, e0 = equilibrate(model, walkers, nw, key, block)
print(f"chain {chain}: equilibrated E = {e0:.6f} ({time.time() - t0:.0f} s)", flush=True)
res = {k: [] for k in ("E", "ad_martin", "ad_zero", "Ep", "Em")}
for i in range(nwin):
    t1 = time.time()
    args = (d, w, e0, jax.random.fold_in(key, i), nblk, block)
    E, a_mart = model.energy_and_derivative(0.0, *args, "martin")
    _, a_zero = model.energy_and_derivative(0.0, *args, "zero")
    ep = [model.window(h, *args, "zero")[0] for h in hs]
    em = [model.window(-h, *args, "zero")[0] for h in hs]
    for k, v in zip(res, (E, a_mart, a_zero, np.array(ep), np.array(em))):
        res[k].append(np.asarray(v))
    _, d, w = model.window(0.0, *args, "zero")
    if i < 2 or i % 10 == 0:
        print(f"chain {chain} window {i}: ({time.time() - t1:.0f} s)", flush=True)
    if i % 5 == 4 or i == nwin - 1:
        np.savez(out, hs=np.array(hs), nw=nw, block=block, **{k: np.array(v) for k, v in res.items()})
print(f"chain {chain} done", flush=True)
