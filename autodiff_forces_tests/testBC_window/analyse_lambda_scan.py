"""Slope of the normal AFQMC energy E(lam) (H1 scaled by 1 + lam, no derivative) from run_lambda_scan.sh, compared with
FCI in the same Hamiltonian (fci/fci_e1.py, fci/fci_lambda.py).
For PART = h2 the scan has L -> (1 + lam/2) L and the slope is compared with the FCI E2.
usage: analyse_lambda_scan.py SCAN_DIR FCI_LAMBDA_LOG [N_DROP] [PART]"""
import sys, glob, re
import numpy as np

scan, fci_log = sys.argv[1:3]
ndrop = int(sys.argv[3]) if len(sys.argv) > 3 else 1000      # blocks dropped (equilibration)
part = sys.argv[4] if len(sys.argv) > 4 else "h1"
E0_FCI = 107.918228                                           # fci_e1.py
E1_FCI = 104.982744 if part == "h1" else 2.935484               # <Psi0|H1|Psi0> or <Psi0|H2|Psi0>

lams, E, S = [], [], []
for f in sorted(glob.glob(f"{scan}/scan_lam*_s1.npz"), key=lambda f: float(re.search(r"scan_lam([-0-9.]+)_s1", f).group(1))):
    lam = float(re.search(r"scan_lam([-0-9.]+)_s1", f).group(1))
    m = []
    for g in sorted(glob.glob(f.replace("_s1.npz", "_s*.npz"))):
        x = np.load(g)["E"].real[ndrop:]
        m.append(x.mean())
    lams.append(lam); E.append(np.mean(m)); S.append(np.std(m, ddof=1) / np.sqrt(len(m)))
lams, E, S = map(np.array, (lams, E, S))
fci = {0.0: E0_FCI}
for line in open(fci_log):
    a = line.split()
    fci[float(a[1])] = float(a[3])

print("lam     E_AFQMC          E_FCI        AFQMC - FCI")
for l, e, s in zip(lams, E, S):
    f = fci.get(round(l, 2))
    print(f"{l:+.2f}  {e:10.4f}({1e4 * s:3.0f})  {f:11.5f}   {e - f:+.3f}({1e3 * s:2.0f})" if f else f"{l:+.2f}  {e:10.4f}")
for h in (0.1, 0.2, 0.3):
    ip, im = np.argmin(abs(lams - h)), np.argmin(abs(lams + h))
    d = (E[ip] - E[im]) / (2 * h); err = np.hypot(S[ip], S[im]) / (2 * h)
    df = (fci[round(h, 2)] - fci[round(-h, 2)]) / (2 * h)
    print(f"central difference h = {h}: AFQMC {d:.3f}({1e3 * err:.0f})   FCI {df:.3f}")
for deg in (3, 4):
    c, cov = np.polyfit(lams, E, deg, w=1 / S, cov="unscaled")
    lf = np.array(sorted(fci)); cf = np.polyfit(lf, [fci[k] for k in lf], deg)
    print(f"polynomial fit degree {deg}: slope at 0  AFQMC {c[-2]:.3f}({1e3 * np.sqrt(cov[-2, -2]):.0f})   FCI {cf[-2]:.3f}")
print(f"FCI <Psi0|{part.upper()}|Psi0> = {E1_FCI}")
