"""Exported Hamiltonians at the central geometry (original export and d0) and with atom 2 moved by +-0.01 A (p01, m01),
all from the central orbitals: energies of afqmc.yaml, H1, H2 (raw, G-resolved)."""
import re, numpy as np
C = "../C_prim_gamma/vasp"
runs = {"central": C, "d0": "d0", "p01": "p01", "m01": "m01"}
def yaml(d):
    t = open(f"{d}/afqmc.yaml", "rb").read().decode("latin1")
    out = {}
    for key in ("PSCENC", "TEWEN", "E1", "DENC", "EXHF", "EATOM"):
        m = re.search(key + r":\s*([-0-9.E+]+)", t); out[key] = float(m.group(1))
    m = re.search(r"PAW double counting: \[([-0-9.E+]+), ([-0-9.E+]+)\]", t); out["PAWdc"] = float(m.group(1)) + float(m.group(2))
    return out
Y = {k: yaml(d) for k, d in runs.items()}
print("term       " + "".join(f"{k:>18s}" for k in runs))
for key in Y["central"]:
    print(f"{key:10s} " + "".join(f"{Y[k][key]:18.10f}" for k in runs))
H1 = {k: np.load(f"{d}/H1.npy") for k, d in runs.items()}
H2 = {k: np.load(f"{d}/H2.npy", mmap_mode="r") for k, d in runs.items()}
print("H1 shape", H1["d0"].shape, "H2 shape", H2["d0"].shape, H2["d0"].dtype)
rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
print("control  d0 vs central: H1 %.2e  H2 %.2e" % (rel(H1["d0"], H1["central"]), rel(np.asarray(H2["d0"]), np.asarray(H2["central"]))))
for k in ("p01", "m01"):
    print(f"{k} vs d0: H1 max|diff| {np.max(np.abs(H1[k]-H1['d0'])):.3e} (max|H1| {np.max(np.abs(H1['d0'])):.3e})   "
          f"H2 max|diff| {np.max(np.abs(np.asarray(H2[k])-np.asarray(H2['d0']))):.3e} (max|H2| {np.max(np.abs(np.asarray(H2['d0']))):.3e})")
# linearity: (p - 0) + (m - 0) should be O(delta^2), (p - m)/2 is the O(delta) part
for name, A in (("H1", H1), ("H2", {k: np.asarray(v) for k, v in H2.items()})):
    odd = (A["p01"] - A["m01"]) / 2; even = (A["p01"] + A["m01"]) / 2 - A["d0"]
    print(f"{name}: |odd part| {np.max(np.abs(odd)):.3e}   |even part| {np.max(np.abs(even)):.3e}   ratio even/odd {np.max(np.abs(even))/np.max(np.abs(odd)):.3e}")
