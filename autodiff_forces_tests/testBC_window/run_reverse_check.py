"""Reverse mode (plain and with checkpointing) against forward mode, Martin's rule, on the windows of run_eq_window.py.
Same chain, seeds, equilibration and window sequence as run_eq_window.py, so the forward values can also be compared
with the stored ones. Per window and perturbation (H1 -> (1 + lam) H1 for E1, L -> (1 + lam/2) L for E2):
  forward:  dE_b/dlam for all blocks b (one jvp pass)
  reverse:  dE_b/dlam for the blocks in SEL (one vjp per selected block) and d(mean_b E_b)/dlam (one vjp),
            plain (all intermediates of every step stored) and with checkpointing (only the walkers of every step
            stored, the inside of the step recomputed in the backward pass)
usage: run_reverse_check.py CHAIN NW N_EQ_BLOCKS N_WINDOWS N_BLOCKS BLOCK OUT.npz"""
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

chain, nw, neq, nwin, nblk, block = (int(a) for a in sys.argv[1:7]); out = sys.argv[7]
SEL = np.array([b for b in (1, 2, 5, 10, 20, 30, 40) if b <= nblk]) - 1        # selected blocks (0-based)
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

# ---- equilibration exactly as run_eq_window.py (the energies are not needed, only the final walkers) ----
key = jax.random.PRNGKey(700 + chain)
eqE = []
for c in range(neq // 10):
    kc = jax.random.fold_in(key, 10**6 + c)
    E, d, w = m1.window(0.0, d, w, e0, kc, 10, block, "zero")
    eqE += list(np.asarray(E).real)
    e0 = float(np.mean(eqE[len(eqE) // 2:]))
print(f"chain {chain}: {neq * block} equilibration steps, E = {e0:.4f}", flush=True)


def reverse(m, args, ckpt):
    """Reverse mode: dE_b/dlam for the selected blocks and d(mean E)/dlam, from one forward pass and its vjp."""
    f = lambda l: m.window(l, *args, "martin", ckpt)[0]
    E, vjp = jax.vjp(f, jnp.asarray(0.0, dtype=m.rtype))
    eye = jnp.eye(E.shape[0], dtype=E.dtype)
    g = jnp.array([vjp(eye[b])[0] for b in SEL])
    gmean = vjp(jnp.full(E.shape, 1.0 / E.shape[0], dtype=E.dtype))[0]
    return g, gmean


def memory(m, args, ckpt):
    """Temporary memory of the compiled reverse pass (bytes) from XLA's memory analysis."""
    f = lambda l: jnp.mean(m.window(l, *args, "martin", ckpt)[0])
    c = jax.jit(jax.grad(f)).lower(jnp.asarray(0.0, dtype=m.rtype)).compile().memory_analysis()
    return None if c is None else int(c.temp_size_in_bytes)


names = [f"{p}_{k}" for p in ("E1", "E2") for k in ("fwd", "rev", "revck", "fwdmean", "revmean", "revckmean")]
res = {k: [] for k in names}; tim = {k: [] for k in ("fwd", "rev", "revck")}
mem = {}
t0 = time.time()
for i in range(nwin):
    k = jax.random.fold_in(key, i)
    args = (d, w, e0, k, nblk, block)
    if i == 0:
        for p, m in (("E1", m1), ("E2", m2)):
            mem[p] = (memory(m, args, False), memory(m, args, True))
            print(f"  memory of the reverse pass for {p}: plain {mem[p][0] / 2**20:.1f} MiB, "
                  f"checkpointed {mem[p][1] / 2**20:.1f} MiB", flush=True)
    for p, m in (("E1", m1), ("E2", m2)):
        ta = time.time(); _, gf = m.energy_and_derivative(0.0, *args, "martin"); gf = np.asarray(gf).real
        tb = time.time(); gr, grm = reverse(m, args, False); gr = np.asarray(gr).real
        tc = time.time(); gc, gcm = reverse(m, args, True); gc = np.asarray(gc).real
        td = time.time()
        if i > 0:                                         # first window includes compilation
            tim["fwd"].append(tb - ta); tim["rev"].append(tc - tb); tim["revck"].append(td - tc)
        res[f"{p}_fwd"].append(gf); res[f"{p}_rev"].append(gr); res[f"{p}_revck"].append(gc)
        res[f"{p}_fwdmean"].append(gf.mean()); res[f"{p}_revmean"].append(float(np.real(grm)))
        res[f"{p}_revckmean"].append(float(np.real(gcm)))
    _, d, w = m1.window(0.0, *args, "zero")
    dr = max(np.max(np.abs(res[f"{p}_rev"][-1] - res[f"{p}_fwd"][-1][SEL])) for p in ("E1", "E2"))
    dc = max(np.max(np.abs(res[f"{p}_revck"][-1] - res[f"{p}_fwd"][-1][SEL])) for p in ("E1", "E2"))
    print(f"chain {chain} window {i + 1}: E1 at end fwd {res['E1_fwd'][-1][-1]:.6f} rev {res['E1_rev'][-1][-1]:.6f} "
          f"revck {res['E1_revck'][-1][-1]:.6f}  max|rev-fwd| {dr:.1e} max|revck-fwd| {dc:.1e}  "
          f"({time.time() - t0:.0f} s)", flush=True)
    np.savez(out, block=block, nw=nw, sel=SEL, mem_E1=np.array(mem["E1"], dtype=float),
             mem_E2=np.array(mem["E2"], dtype=float), **{f"t_{k}": np.array(v) for k, v in tim.items()},
             **{n: np.array(v) for n, v in res.items()})
print(f"chain {chain} done; time per window and perturbation: " +
      "  ".join(f"{k} {np.mean(v):.1f} s" for k, v in tim.items() if v), flush=True)
