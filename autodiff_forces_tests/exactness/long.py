import sys, time
import numpy as np
sys.argv = [sys.argv[0], "gamma"]
exec(open("check.py").read().split("name = sys.argv[1]")[0])
import jax, jax.numpy as jnp
config, H, trial, walkers, model = build("gamma", 100)
d, w = walkers.slater_det, walkers.weights
e0 = float(new.measure_energy(config, trial, walkers, H)[0].real)
nb, bl = 40, 10            # window of 400 steps, rebalancing every 10 steps
for win in range(3):
    key = jax.random.PRNGKey(500 + win)
    args = (d, w, e0, key, nb, bl, "martin")
    t = time.time(); E, dE = model.energy_and_derivative(0.0, *args); dE.block_until_ready(); tf = time.time() - t
    last = lambda l: model.window(l, *args)[0][-1]
    t = time.time(); g = jax.grad(last)(0.0); tr = time.time() - t
    t = time.time(); gc = jax.grad(lambda l: model.window(l, d, w, e0, key, nb, bl, "martin", True)[0][-1])(0.0); tc = time.time() - t
    print(f"window {win+1}: after 40/100/200/400 steps forward = "
          + ", ".join(f"{float(dE[k-1]):.6f}" for k in (4, 10, 20, 40))
          + f" | after 400 steps: forward {float(dE[-1]):.12f}  reverse {float(g):.12f}  reverse+ckpt {float(gc):.12f}"
          + f"  rel.diff {abs(float(dE[-1]) - float(g)) / abs(float(g)):.1e}  (time s: fwd {tf:.0f}, rev {tr:.0f}, ckpt {tc:.0f})")
    _, d, w = model.window(0.0, *args)
