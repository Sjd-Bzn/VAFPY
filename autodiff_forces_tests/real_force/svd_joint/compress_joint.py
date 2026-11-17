"""Joint SVD compression of L and dL/dx at r003 (Gamma, one sector): one fixed set of columns for both.
Rows of M are the orbital pairs (b1, b2), columns the G vectors (as in compress_H2_compact.py). The right singular
vectors W of [M(c0); s dM] (s = |M| / |dM| so that both enter with the same weight) with s_i > threshold * s_max
define the compressed columns: Lc = M(c0) W, dLc = dM W. Since every term of the energy and the propagator is a
sum over g of L L^+, a common W keeps L L^+, L dL^+ and dL dL^+ to the truncation error."""
import sys
import numpy as np
d = 0.01
nb = 16

def matrix(path):                                         # (g, b1, b2) -> M[(b1, b2), g]
    raw = np.load(f"{path}/H2.npy")
    return np.ascontiguousarray(raw.transpose(1, 2, 0)).reshape(nb * nb, -1)

def compact(Mc):                                          # M[(b1,b2), n] -> Lc[b2, b1, n]
    return np.ascontiguousarray(Mc.reshape(nb, nb, -1).transpose(1, 0, 2))

M = {t: matrix(f"../{t}") for t in ("c0", "p01", "m01")}
dM = (M["p01"] - M["m01"]) / (2 * d)
s = np.linalg.norm(M["c0"]) / np.linalg.norm(dM)
_, sv, Wh = np.linalg.svd(np.vstack([M["c0"], s * dM]), full_matrices=False)
for thr in [float(x) for x in sys.argv[1:]] or [1e-4]:
    keep = sv > thr * sv[0]
    W = Wh[keep].conj().T                                 # (ng, n)
    n = int(keep.sum())
    Lc, dLc = M["c0"] @ W, dM @ W
    err = lambda A, B: np.linalg.norm(A @ W @ W.conj().T @ B.conj().T - A @ B.conj().T) / np.linalg.norm(A @ B.conj().T)
    print(f"threshold {thr:.0e}: kept {n:4d} of {M['c0'].shape[1]}   rel.err  LL+ {err(M['c0'], M['c0']):.1e}"
          f"  L dL+ {err(M['c0'], dM):.1e}  dL dL+ {err(dM, dM):.1e}")
    tag = f"{thr:.0e}"
    np.save(f"H2_zip_{tag}.npy", compact(Lc)); np.save(f"dH2_zip_{tag}.npy", compact(dLc))
    np.save(f"Q_sizes_{tag}.npy", np.array([n], dtype=np.int64))
h1 = {t: np.load(f"../{t}/H1.npy") for t in ("c0", "p01", "m01")}
np.save("H1_c0.npy", h1["c0"]); np.save("dH1.npy", (h1["p01"] - h1["m01"]) / (2 * d))
