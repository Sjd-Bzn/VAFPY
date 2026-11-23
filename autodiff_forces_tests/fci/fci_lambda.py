"""FCI energies with H1 scaled by (1 + lam) (default) or with L scaled by (1 + lam/2) (first argument h2), for
comparison with the AFQMC lambda scans.
usage: fci_lambda.py [h2] LAM [LAM ...]"""
import sys
import numpy as np
from pyscf import fci, ao2mo, lib

lib.num_threads(1)
P = "/fs/home/sajad/projects/data/test/diamond_gamma_prim"
norb, nelec = 16, (4, 4)
h = np.load(f"{P}/H1_svd.npy")[:, :, 0].real.copy()
L = np.load(f"{P}/H2_zip.npy")
eri = ao2mo.restore(8, np.einsum("pqg,rsg->pqrs", L, L.conj()).real.copy(), norb)
s = fci.direct_spin1.FCI(); s.conv_tol, s.max_cycle = 1e-12, 200
ci = None
h2 = sys.argv[1] == "h2"
for lam in map(float, sys.argv[2:] if h2 else sys.argv[1:]):
    e, ci = s.kernel(h, (1 + lam / 2) ** 2 * eri, norb, nelec, ci0=ci) if h2 else s.kernel((1 + lam) * h, eri, norb, nelec, ci0=ci)
    print(f"lam {lam:+.2f}  E_FCI {e:.6f}", flush=True)
