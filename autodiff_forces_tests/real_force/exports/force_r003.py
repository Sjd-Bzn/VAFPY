"""Frozen-orbital force at r003 (atom 2 at +0.03 A along x): exports at 0.03 (c0), 0.04 (p01), 0.02 (m01) A, all from
the r003 orbitals, against the HF force of VASP at r003."""
import re, numpy as np
R = "r003"; d = 0.01  # A
runs = {"vasp": f"{R}/vasp", "c0": f"{R}/c0", "p01": f"{R}/p01", "m01": f"{R}/m01"}
def yaml(p):
    t = open(f"{p}/afqmc.yaml", "rb").read().decode("latin1"); out = {}
    for key in ("PSCENC", "TEWEN", "E1", "DENC", "EXHF", "EATOM"):
        out[key] = float(re.search(key + r":\s*([-0-9.E+]+)", t).group(1))
    m = re.search(r"PAW double counting: \[([-0-9.E+]+), ([-0-9.E+]+)\]", t); out["PAWdc"] = float(m.group(1)) + float(m.group(2))
    return out
Y = {k: yaml(p) for k, p in runs.items()}
print("term       " + "".join(f"{k:>17s}" for k in runs) + "     -dS/dx (eV/A)")
S = {k: 0.0 for k in runs}
for key in ("E1", "DENC", "EXHF", "TEWEN", "PAWdc", "PSCENC", "EATOM"):
    f = -(Y["p01"][key] - Y["m01"][key]) / (2 * d)
    print(f"{key:10s} " + "".join(f"{Y[k][key]:17.10f}" for k in runs) + f"  {f:14.6f}")
    if key not in ("PSCENC", "EATOM"):
        for k in runs: S[k] += Y[k][key]
F = -(S["p01"] - S["m01"]) / (2 * d)
curv = (S["p01"] + S["m01"] - 2 * S["c0"]) / d**2
print(f"S = E1+DENC+EXHF+TEWEN+PAWdc: c0 {S['c0']:.10f}  p01 {S['p01']:.10f}  m01 {S['m01']:.10f}")
print(f"frozen-orbital force on atom 2, x:  {F:.6f} eV/A   (curvature {curv:.3f} eV/A^2)")
t = open(f"{R}/vasp/OUTCAR.2_hf-4").read()
blk = t[t.rfind("TOTAL-FORCE"):].splitlines()[2:4]
print("VASP HF force (OUTCAR.2_hf-4):"); [print("  ", l) for l in blk]
Fv = float(blk[1].split()[3]); print(f"difference {F - Fv:+.6f} eV/A  ({(F - Fv)/Fv*100:+.3f} %)")
H1 = {k: np.load(f"{p}/H1.npy") for k, p in runs.items()}
H2 = {k: np.load(f"{p}/H2.npy") for k, p in runs.items()}
rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
print("control c0 vs vasp export: H1 %.2e  H2 %.2e" % (rel(H1["c0"], H1["vasp"]), rel(H2["c0"], H2["vasp"])))
for name, A in (("H1", H1), ("H2", H2)):
    odd = (A["p01"] - A["m01"]) / 2; even = (A["p01"] + A["m01"]) / 2 - A["c0"]
    print(f"{name}: |odd| {np.max(np.abs(odd)):.3e}  |even| {np.max(np.abs(even)):.3e}  even/odd {np.max(np.abs(even))/np.max(np.abs(odd)):.3e}")
# HF energy from the matrices (4 occupied orbitals, closed shell): E1 = 2 sum_i h_ii ; check against yaml
for k in ("c0", "p01", "m01"):
    h = H1[k][..., 0]; L = H2[k]; o = slice(0, 4)
    e1 = 2 * np.trace(h[o, o]).real
    J = 2 * np.sum(np.abs(np.einsum("gii->g", L[:, o, o]))**2)  # 4 * (1/2) sum_g |sum_i L_g,ii|^2
    K = np.sum(np.abs(L[:, o, o])**2)                         # sum_g sum_ij |L_g,ij|^2
    print(f"{k}: E1 from H1 {e1:.10f} (yaml {Y[k]['E1']:.10f});  J {J:.6f} (DENC {Y[k]['DENC']:.6f});  -K {-K:.6f} (EXHF {Y[k]['EXHF']:.6f})")
