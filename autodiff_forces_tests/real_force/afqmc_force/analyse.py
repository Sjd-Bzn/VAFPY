"""dE/dx (eV/A) from AD (Martin, zero rule) and the correlated FD, per window length; force on atom 2 along x."""
import sys, glob
import numpy as np
from common import DE_HF, DX_CLASSICAL
pat = sys.argv[1] if len(sys.argv) > 1 else "pilot_*.npz"
Z = [np.load(f) for f in sorted(glob.glob(pat))]
hs, block = Z[0]["hs"], int(Z[0]["block"])
cat = lambda k: np.concatenate([z[k] for z in Z])
am, az, Ep, Em, E = cat("ad_martin"), cat("ad_zero"), cat("Ep"), cat("Em"), cat("E")
N = am.shape[0]
fd = (Ep - Em) / (2 * hs[None, :, None])
print(f"{N} windows ({len(Z)} chains); HF dE/dx = {DE_HF:.4f}; classical d(Ewald + PAWdc)/dx = {DX_CLASSICAL:.4f}")
def me(x, nb=10):                                        # mean and blocked error (consecutive windows per chain)
    x = np.asarray(x); m = x.mean()
    blocks = np.array_split(x, min(nb * len(Z), len(x)))
    bm = np.array([b.mean() for b in blocks])
    return m, bm.std(ddof=1) / np.sqrt(len(bm)), x.std(ddof=1)
for L in (40, 100, 200, 400):
    b = L // block - 1
    if b >= am.shape[1]:
        continue
    mE, eE, _ = me(E[:, b]); mm, em, sm = me(am[:, b]); mz, ez, _ = me(az[:, b])
    line = f"L={L:3d}: E {mE:.4f}({eE:.4f})  AD Martin {mm:.4f}({em:.4f}) [spread {sm:.3f}]  zero {mz:.4f}({ez:.4f})"
    for j, h in enumerate(hs):
        mf, ef, sf = me(fd[:, j, b]); dm, de, _ = me(fd[:, j, b] - am[:, b])
        line += f"\n        FD h={h:.3f} A {mf:.4f}({ef:.4f}) [spread {sf:.3f}]   FD - Martin {dm:+.4f}({de:.4f})"
    print(line)
    print(f"        force on atom 2 (Martin): F_x = -(dE/dx + {DX_CLASSICAL:.4f}) = {-(mm + DX_CLASSICAL):.4f}({em:.4f}) eV/A"
          f"   [HF {-(DE_HF + DX_CLASSICAL):.4f}]")
