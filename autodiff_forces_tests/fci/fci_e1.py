"""Exact (FCI, pyscf) reference for stage 1 in the 16-orbital Hamiltonian of the AFQMC tests (diamond, Gamma point).

(pq|rs) = sum_g L[p,q,g] L*[r,s,g] from H2_zip.npy reproduces VASP's E1, Hartree and EXHF of the HF determinant
exactly. Prints for the Sz = 0 ground state Psi0 (4 + 4 electrons in 16 orbitals):
  E0                                     exact ground-state energy (compare with the AFQMC energy)
  <H1>_pure  = <Psi0|H1|Psi0>            exact one-body energy (= dE0/dlam by Hellmann-Feynman)
  <H1>_mixed = <HF|H1|Psi0>/<HF|Psi0>    what normal AFQMC measures as E1 with the HF trial, for exact walkers
  dE0/dlam by a central difference of FCI energies at H1 -> (1 +- h) H1 (check of Hellmann-Feynman)
usage: fci_e1.py [H] [NTHREADS]
"""
import sys, time
import numpy as np
from pyscf import fci, ao2mo, lib

P = "/fs/home/sajad/projects/data/test/diamond_gamma_prim"
hfd = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
if len(sys.argv) > 2:
    lib.num_threads(int(sys.argv[2]))
norb, nelec = 16, (4, 4)
h = np.load(f"{P}/H1_svd.npy")[:, :, 0]
L = np.load(f"{P}/H2_zip.npy")
eri = np.einsum("pqg,rsg->pqrs", L, L.conj())
assert np.abs(h.imag).max() < 1e-10 and np.abs(eri.imag).max() < 1e-10
h, eri = h.real.copy(), ao2mo.restore(8, eri.real.copy(), norb)

t0 = time.time()
solver = fci.direct_spin1.FCI()
solver.conv_tol, solver.max_cycle = 1e-12, 200
na = fci.cistring.num_strings(norb, nelec[0])
hf = np.zeros((na, na)); hf[0, 0] = 1.0            # string 0 = lowest 4 orbitals = HF determinant
e_hf = solver.energy(h, eri, hf, norb, nelec)
e1_hf = float(np.sum(h * solver.make_rdm1(hf, norb, nelec)))
print(f"{na * na} determinants; HF: E {e_hf:.10f}  E1 {e1_hf:.10f}  E2 {e_hf - e1_hf:.10f}", flush=True)

E0, ci = solver.kernel(h, eri, norb, nelec, ci0=hf)
if ci[0, 0] < 0:
    ci = -ci
g = solver.make_rdm1(ci, norb, nelec)
e1_pure = float(np.sum(h * g))
t = solver.trans_rdm1(hf, ci, norb, nelec)          # t[p,q] = <HF| p+ q |Psi0>
e1_mixed = float(np.sum(h * t) / ci[0, 0])
print(f"FCI: E0 {E0:.8f}   |<HF|Psi0>|^2 {ci[0, 0]**2:.5f}   ({time.time() - t0:.0f} s)")
print(f"  <H1> pure  <Psi0|H1|Psi0>          {e1_pure:.6f}    <H2> pure  {E0 - e1_pure:.6f}")
print(f"  <H1> mixed <HF|H1|Psi0>/<HF|Psi0>  {e1_mixed:.6f}    <H2> mixed {E0 - e1_mixed:.6f}", flush=True)
ep = solver.kernel((1 + hfd) * h, eri, norb, nelec, ci0=ci)[0]
em = solver.kernel((1 - hfd) * h, eri, norb, nelec, ci0=ci)[0]
print(f"  dE0/dlam, central difference of FCI energies (h = {hfd}): {(ep - em) / (2 * hfd):.6f}   ({time.time() - t0:.0f} s)")
