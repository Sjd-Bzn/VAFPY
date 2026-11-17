"""Test F: memory and time of the derivative against the window length (one configuration per process, so the
peak device memory belongs to this configuration only).
  mode fwd  : forward mode (jvp), dE/dlam after every block
  mode rev  : reverse mode, d E_last / dlam (stores the trajectory)
  mode ckpt : reverse mode with checkpointing of every step
Memory: XLA's estimate of the compiled program (temporary buffers), the measured peak on the GPU, and on the CPU the
peak resident memory of the process (includes the Python/JAX baseline, given separately as rss_base_bytes).
Time: after compilation and one warm-up call, mean of NREP calls (env NREP, default 3).
usage: run_mem.py SYSTEM(gamma|k222) NW L MODE(plain|fwd|rev|ckpt) [PREC(Double|Single)]
"""
import os, sys, time, json, resource
import numpy as np
SRC = os.environ.get("VAFPY_SRC", "/fs/home/sajad/projects/git/AFQMC-autodiff-forces/source/vafpy_v3")
sys.path.insert(0, SRC)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functions as new
import autodiff_forces as af
from test_compact_hamiltonian import _DATA_ROOT, COMPACT, make_config, perturbed

system, nw, L, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
prec = sys.argv[5] if len(sys.argv) > 5 else "Double"
path, kw = {"gamma": (os.path.join(_DATA_ROOT, "test", "diamond_gamma_prim"), dict(nk=1, nb=16, ne=4, ng=136)),
            "k222": (COMPACT, dict())}[system]
config = make_config(nw, precision=prec, backend=new.JaxBackend(5), **kw)
q = new.obtain_Q_list(config, os.path.join(path, "Q_list.npy"))
h1 = new.obtain_H1(config, os.path.join(path, "H1_svd.npy"))
h2c, _ = new.obtain_H2_host(config, os.path.join(path, "H2_zip.npy"))
sizes = new.obtain_Q_sizes(config, os.path.join(path, "Q_sizes.npy"))
H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
trial, walkers = new.initialize_determinant(config)
H.setup_energy_expressions(config, trial)
model = af.Model(config, H, trial, V=np.asarray(h1))
walkers = perturbed(config, walkers)
d, w = jnp.asarray(walkers.slater_det, dtype=model.ctype), jnp.asarray(walkers.weights, dtype=model.ctype)
e0 = float(new.measure_energy(config, trial, walkers, H)[0].real)
key = jax.random.PRNGKey(1)
rss_base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
block = 10
nb = L // block

def energy(lam):
    return model.window(lam, d, w, e0, key, nb, block, "martin", mode == "ckpt")[0]

if mode == "plain":
    f = jax.jit(lambda lam: energy(lam))
elif mode == "fwd":
    f = jax.jit(lambda lam: jax.jvp(energy, (lam,), (jnp.ones_like(lam),)))
else:
    f = jax.jit(jax.value_and_grad(lambda lam: energy(lam)[-1]))
lam = jnp.asarray(0.0, dtype=model.rtype)
t = time.time(); comp = f.lower(lam).compile(); tc = time.time() - t
ma = comp.memory_analysis()
est = {k: getattr(ma, k, None) for k in ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes")}
out = comp(lam); jax.block_until_ready(out)
ts = []
for _ in range(int(os.environ.get("NREP", 3))):
    t = time.time(); out = comp(lam); jax.block_until_ready(out); ts.append(time.time() - t)
dev = jax.devices()[0]
stats = dev.memory_stats() or {}
res = dict(system=system, nw=nw, L=L, mode=mode, prec=prec, device=str(dev), compile_s=tc, time_s=float(np.mean(ts)),
           peak_bytes=stats.get("peak_bytes_in_use"), rss_base_bytes=rss_base,
           rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, **est)
print(json.dumps(res), flush=True)
