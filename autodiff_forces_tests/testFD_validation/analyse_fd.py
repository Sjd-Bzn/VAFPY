"""Ensemble finite-difference check: per window D_x = x - AD(zero rule), mean and error over windows,
for x = AD(Martin) and FD(h) = (E_L(+h) - E_L(-h)) / 2h; per-window spread; windows needed for 0.05 eV."""
import sys, glob
import numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else "pilot_fd_*.npz"
Z = [np.load(f) for f in sorted(glob.glob(pat))]
hs = Z[0]["hs"]; block = int(Z[0]["block"])
cat = lambda k: np.concatenate([z[k] for z in Z])
am, az, Ep, Em = cat("ad_martin"), cat("ad_zero"), cat("Ep"), cat("Em")
N = am.shape[0]
fd = (Ep - Em) / (2 * hs[None, :, None])            # (window, h, block)
print(f"{N} windows, L = block x {block} steps")
def ms(x): return x.mean(), x.std(ddof=1), x.std(ddof=1) / np.sqrt(len(x))
for L in (10, 40, 100, 200, 400):
    b = L // block - 1
    if b >= am.shape[1]: continue
    m, s, e = ms(am[:, b] - az[:, b])
    print(f"L={L:4d}  zero {az[:, b].mean():9.4f}  Martin-zero {m:+.4f}({e:.4f}) sd {s:.3f}")
    for j, h in enumerate(hs):
        m, s, e = ms(fd[:, j, b] - az[:, b])
        mf, sf, ef = ms(fd[:, j, b])
        need = (s / 0.05) ** 2
        jump = np.mean(np.abs(fd[:, j, b] - az[:, b]) > 1e-3)
        print(f"         h={h:<7g} FD {mf:9.4f}({ef:.4f})  FD-zero {m:+.4f}({e:.4f}) sd {s:6.3f}  "
              f"windows for 0.05 eV: {need:7.0f}   windows with comb jump {100*jump:5.1f}%")
