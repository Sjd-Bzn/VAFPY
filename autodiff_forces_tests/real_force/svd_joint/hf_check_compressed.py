"""HF energy and dE/dx (walkers = trial) from the jointly compressed L, dL against the raw values
(hf_check_dHdR.py: E = 113.6480608517 eV, dE/dx = 0.86412401 eV/A)."""
import sys
import numpy as np
V3 = "/fs/home/sajad/projects/git/AFQMC-autodiff-forces/source/vafpy_v3"
sys.path.insert(0, V3)
import jax
jax.config.update("jax_enable_x64", True)
import functions as new
import autodiff_forces as af
from test_compact_hamiltonian import make_config
E_RAW, DE_RAW = 113.6480608517, 0.86412401
for tag in ("1e-03", "1e-04", "1e-05", "1e-06"):
    Lc, dL = np.load(f"H2_zip_{tag}.npy"), np.load(f"dH2_zip_{tag}.npy")
    ng = Lc.shape[2]
    config = make_config(1, precision="Double", backend=new.JaxBackend(5), nk=1, nb=16, ne=4, ng=ng)
    q = new.obtain_Q_list(config, "/fs/home/sajad/projects/data/test/diamond_gamma_prim/Q_list.npy")
    H = new.build_hamiltonian(config, new.obtain_H1(config, "H1_c0.npy"), Lc, "compact", q)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    dh1 = np.asarray(new.obtain_H1(config, "dH1.npy"))
    model = af.Model(config, H, trial, V=dh1, L=Lc, dL=dL)
    e, de = jax.value_and_grad(lambda l: model.mixed_energy(l, walkers.slater_det, walkers.weights))(0.0)
    print(f"{tag}: ng {ng:4d}   E {float(e):.10f} ({float(e) - E_RAW:+.1e})   dE/dx {float(de):.8f} ({float(de) - DE_RAW:+.1e})")
