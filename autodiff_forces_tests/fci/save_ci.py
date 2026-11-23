"""FCI ground state of the 16-orbital Hamiltonian, saved as a trial for AFQMC (CI matrix C[alpha string, beta string]
and the occupied orbitals of every string, pyscf string order).
usage: save_ci.py OUT.npz"""
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
na = fci.cistring.num_strings(norb, 4)
hf = np.zeros((na, na)); hf[0, 0] = 1.0
e, ci = s.kernel(h, eri, norb, nelec, ci0=hf)
ci = ci if ci[0, 0] > 0 else -ci
strings = fci.cistring.make_strings(range(norb), 4)
occ = np.array([[i for i in range(norb) if (int(st) >> i) & 1] for st in strings])
e1 = float(np.sum(h * s.make_rdm1(ci, norb, nelec)))
np.savez(sys.argv[1], ci=ci, occ=occ, e0=e, e1_pure=e1)
print(f"E0 {e:.8f}  <H1> {e1:.6f}  saved {sys.argv[1]}  occ[0] {occ[0]}")
