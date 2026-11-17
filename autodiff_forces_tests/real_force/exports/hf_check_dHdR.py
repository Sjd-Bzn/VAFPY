"""Derivative code with the real dH/dR at r003 (atom 2 at +0.03 A along x), walkers = HF trial.
dH1 = (H1(+0.01) - H1(-0.01)) / 0.02 A, dL = (L(+0.01) - L(-0.01)) / 0.02 A, raw (uncompressed) L, 6422 G vectors,
all from the r003 orbitals. Expected: dE/dx = d(E1 + DENC + EXHF)/dx of afqmc.yaml."""
import os, re, sys
import numpy as np
V3 = "/fs/home/sajad/projects/git/AFQMC-autodiff-forces/source/vafpy_v3"
sys.path.insert(0, V3)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import functions as new
import autodiff_forces as af
from test_compact_hamiltonian import make_config

d = 0.01
def yaml(p):
    t = open(f"{p}/afqmc.yaml", "rb").read().decode("latin1")
    return {k: float(re.search(k + r":\s*([-0-9.E+]+)", t).group(1)) for k in ("E1", "DENC", "EXHF")}
def L(p):
    return np.ascontiguousarray(np.load(f"{p}/H2.npy").transpose(2, 1, 0))     # (g, b1, b2) -> Lc[b2, b1, g]

config = make_config(1, precision="Double", backend=new.JaxBackend(5), nk=1, nb=16, ne=4, ng=6422)
q = new.obtain_Q_list(config, "/fs/home/sajad/projects/data/test/diamond_gamma_prim/Q_list.npy")
h1 = {t: np.asarray(new.obtain_H1(config, f"{t}/H1.npy")) for t in ("c0", "p01", "m01")}
Lc = {t: L(t) for t in ("c0", "p01", "m01")}
H = new.build_hamiltonian(config, config.backend.array(h1["c0"]), Lc["c0"], "compact", q)
trial, walkers = new.initialize_determinant(config)
H.setup_energy_expressions(config, trial)
dh1 = (h1["p01"] - h1["m01"]) / (2 * d)
dL = (Lc["p01"] - Lc["m01"]) / (2 * d)
model = af.Model(config, H, trial, V=dh1, L=Lc["c0"], dL=dL)

dets, w = walkers.slater_det, walkers.weights
e, de = jax.value_and_grad(lambda l: model.mixed_energy(l, dets, w))(0.0)
Y = {t: yaml(t) for t in ("c0", "p01", "m01")}
S = {t: sum(Y[t].values()) for t in Y}
print(f"E_HF (electronic) code {float(e):.10f}   yaml E1+DENC+EXHF {S['c0']:.10f}   diff {float(e) - S['c0']:+.2e}")
ref = (S["p01"] - S["m01"]) / (2 * d)
print(f"dE/dx code (AD, dH/dR in)  {float(de):.8f} eV/A")
print(f"dE/dx finite difference    {ref:.8f} eV/A   diff {float(de) - ref:+.2e}")
for k in ("E1", "DENC", "EXHF"):
    print(f"   {k:5s} {(Y['p01'][k] - Y['m01'][k]) / (2 * d):+.6f}")
# one-body part only (dL = 0) and two-body part only
m1 = af.Model(config, H, trial, V=dh1)
print(f"   one-body part (dH1 only) {float(jax.grad(lambda l: m1.mixed_energy(l, dets, w))(0.0)):+.6f}")
m2 = af.Model(config, H, trial, V=0 * dh1, L=Lc["c0"], dL=dL)
print(f"   two-body part (dL only)  {float(jax.grad(lambda l: m2.mixed_energy(l, dets, w))(0.0)):+.6f}")

# exactness: around the midpoint L_mid = (L(+) + L(-))/2 the HF energy is quadratic in lam and E(lam = +-d) are the
# energies of the +-0.01 A Hamiltonians, so the derivative must equal their central difference to rounding
h1m, Lm = (h1["p01"] + h1["m01"]) / 2, (Lc["p01"] + Lc["m01"]) / 2
Hm = new.build_hamiltonian(config, config.backend.array(h1m), Lm, "compact", q)
Hm.setup_energy_expressions(config, trial)
mm = af.Model(config, Hm, trial, V=dh1, L=Lm, dL=dL)
dem = float(jax.grad(lambda l: mm.mixed_energy(l, dets, w))(0.0))
print(f"around the midpoint: dE/dx code {dem:.10f}   finite difference {ref:.10f}   diff {dem - ref:+.2e}")
