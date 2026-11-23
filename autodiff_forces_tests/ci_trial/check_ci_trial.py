"""Checks of ci_trial.ModelCI before any AFQMC run.
1. CI trial = HF determinant only: overlap, force bias and E1 (mixed) equal those of autodiff_forces.Model for
   random (non-HF) walkers.
2. CI trial = FCI ground state, walkers = HF: overlap = C[0,0], E1 (mixed) = <Psi0|H1|HF>/<Psi0|HF> = 101.2236 (FCI).
usage: check_ci_trial.py FCI_CI.npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_trial
from ci_trial import jax, jnp, new, af
jax.config.update("jax_enable_x64", True)
from test_compact_hamiltonian import make_config
from test_autodiff_forces import GAMMA as P

nw = 6
config = make_config(nw, precision="Double", backend=new.JaxBackend(7), nk=1, nb=16, ne=4, ng=136)
q = new.obtain_Q_list(config, os.path.join(P, "Q_list.npy")); h1 = new.obtain_H1(config, os.path.join(P, "H1_svd.npy"))
h2c, _ = new.obtain_H2_host(config, os.path.join(P, "H2_zip.npy")); sizes = new.obtain_Q_sizes(config, os.path.join(P, "Q_sizes.npy"))
H = new.build_hamiltonian(config, h1, h2c, "compact", q, sizes)
trial, walkers = new.initialize_determinant(config); H.setup_energy_expressions(config, trial)
V = np.asarray(h1)
m = af.Model(config, H, trial, V=V)
z = np.load(sys.argv[1])
occ = z["occ"]; ns = len(occ)

rng = np.random.default_rng(1)
d = jnp.asarray(walkers.slater_det, dtype=m.ctype)
d = d + 0.3 * (rng.normal(size=d.shape) + 1j * rng.normal(size=d.shape))     # non-HF walkers
w = jnp.ones(nw, dtype=m.ctype)
P0 = m.pieces(0.0)

C_hf = np.zeros((ns, ns)); C_hf[0, 0] = 1.0
mh = ci_trial.ModelCI(config, H, trial, V, C_hf, occ, 0.0, np.asarray(h2c))
th = m.biorthogonalize(d)
fb_ref = m.k["force_bias"](P0["M"], P0["c"], P0["cc"], th)
fb_ci = mh.force_bias(d, P0)
e1_ref = jax.jvp(lambda l: m.mixed_energy(l, d, w), (0.0,), (1.0,))[1]
e1_ci = jnp.mean(mh.e1_mixed(d))
print("1. HF trial:  max |overlap - ref| / |ref|", float(jnp.max(jnp.abs(mh.overlap(d) / m.overlap(d) - 1))))
print("              max |force bias - ref|     ", float(jnp.max(jnp.abs(fb_ci - fb_ref))), " (max |ref|", float(jnp.max(jnp.abs(fb_ref))), ")")
print("              E1 mixed (equal weights)   ", float(jnp.real(e1_ci)), " ref", float(e1_ref))

mf = ci_trial.ModelCI(config, H, trial, V, z["ci"], occ, float(z["e0"]), np.asarray(h2c))
d0 = jnp.asarray(walkers.slater_det, dtype=m.ctype)
print("2. FCI trial, walkers = HF: overlap", float(jnp.real(mf.overlap(d0)[0])), " C[0,0]", float(z["ci"][0, 0]))
print("              E1 mixed", float(jnp.real(mf.e1_mixed(d0)[0])), " (FCI <Psi0|H1|HF>/<Psi0|HF> = 101.223633)")
print("              FCI <Psi0|H1|Psi0> =", float(z["e1_pure"]), " E0 =", float(z["e0"]))
