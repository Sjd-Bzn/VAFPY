import os, sys, time
import numpy as np
SRC = "/fs/home/sajad/projects/git/AFQMC-autodiff-forces/source/vafpy_v3"
sys.path.insert(0, SRC)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functions as new
import autodiff_forces as af
from test_compact_hamiltonian import _DATA_ROOT, COMPACT, make_config, perturbed

SYSTEMS = {"gamma": (os.path.join(_DATA_ROOT, "test", "diamond_gamma_prim"), dict(nk=1, nb=16, ne=4, ng=136)),
           "k222": (COMPACT, dict())}

def build(name, nw):
    path, kw = SYSTEMS[name]
    config = make_config(nw, precision="Double", backend=new.JaxBackend(5), **kw)
    q = new.obtain_Q_list(config, os.path.join(path, "Q_list.npy"))
    h1 = new.obtain_H1(config, os.path.join(path, "H1_svd.npy"))
    h2c, _ = new.obtain_H2_host(config, os.path.join(path, "H2_zip.npy"))
    sizes = new.obtain_Q_sizes(config, os.path.join(path, "Q_sizes.npy"))
    H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    model = af.Model(config, H, trial, V=np.asarray(h1))
    return config, H, trial, perturbed(config, walkers), model

name = sys.argv[1]
print(f"===== system {name}")
config, H, trial, walkers, model = build(name, 6)

# 1. HF limit
dets = jnp.asarray(np.array([np.asarray(trial)] * 6), dtype=model.ctype)
w = jnp.ones(6, dtype=model.ctype)
e, de = jax.value_and_grad(lambda l: model.mixed_energy(l, dets, w))(0.0)
print(f"[1] HF limit: AD dE/dlam = {float(de):.12f}   exact 2*sum_occ<i|V|i> = {af.hf_derivative(trial, model.V):.12f}"
      f"   diff = {abs(float(de)-af.hf_derivative(trial, model.V)):.1e}")

# 2. forward vs finite difference, no rebalancing
key = jax.random.PRNGKey(7)
args = (walkers.slater_det, walkers.weights, 0.0, key, 3, 4, "off")
E, dE = model.energy_and_derivative(0.0, *args)
h = 1e-4
fd = (model.window(h, *args)[0] - model.window(-h, *args)[0]) / (2 * h)
print("[2] no rebalancing, 3 blocks x 4 steps: forward-mode AD vs finite difference (same random numbers, h=1e-4)")
for b in range(3):
    print(f"    block {b+1}: AD {float(dE[b]):.10f}   FD {float(fd[b]):.10f}   rel.diff {abs(float(dE[b]-fd[b]))/abs(float(fd[b])):.1e}")

# 3. forward vs reverse vs reverse+checkpoint, with rebalancing
print("[3] forward vs reverse mode, rebalancing after every block (lam=0.3)")
for nb, bl, nw in [(3, 3, 6), (10, 10, 20)]:
    if nw != 6:
        config, H, trial, walkers, model = build(name, nw)
    for mode in ["martin", "zero"]:
        args = (walkers.slater_det, walkers.weights, 0.0, jax.random.PRNGKey(11), nb, bl, mode)
        t = time.time(); E, dE = model.energy_and_derivative(0.3, *args); fwd = float(np.mean(dE)); tf = time.time()-t
        t = time.time(); m, g = model.mean_energy_gradient(0.3, *args); tr = time.time()-t
        t = time.time(); mc, gc = model.mean_energy_gradient(0.3, *args, checkpoint=True); tc = time.time()-t
        print(f"    {nb} blocks x {bl} steps, {nw} walkers, {mode:6s}: forward {fwd:.12f}  reverse {float(g):.12f}"
              f"  reverse+ckpt {float(gc):.12f}   |fwd-rev|/|rev| {abs(fwd-float(g))/abs(float(g)):.1e}"
              f"  |rev-ckpt| {abs(float(g)-float(gc)):.1e}")
