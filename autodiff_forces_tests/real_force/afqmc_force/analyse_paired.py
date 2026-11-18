"""Contribution of the population control: (Martin - zero rule) and (finite difference - zero rule) of the same
windows, with errors from blocks of consecutive windows; distribution of the per-window derivative at 400 steps."""
import sys, glob
import numpy as np
pat = sys.argv[1] if len(sys.argv) > 1 else "prod_*.npz"
Z = [np.load(f) for f in sorted(f for p in pat.split(",") for f in glob.glob(p))]
cat = lambda k: np.concatenate([z[k] for z in Z]); hs = Z[0]["hs"]
am, az, Ep, Em = cat("ad_martin"), cat("ad_zero"), cat("Ep"), cat("Em")
fd = (Ep - Em) / (2 * hs[None, :, None])
def me(x, nb=80):
    b = np.array([c.mean() for c in np.array_split(x, nb)]); return x.mean(), b.std(ddof=1) / np.sqrt(nb)
print(f"{am.shape[0]} windows; paired against the zero rule (eV/A)")
for L in (40, 100, 200, 400):
    b = L // 10 - 1
    m = me(am[:, b] - az[:, b]); s = f"L={L:3d}: Martin-zero {m[0]:+.3f}({m[1]:.3f})"
    for j, h in enumerate(hs):
        f = me(fd[:, j, b] - az[:, b]); s += f"  FD{h:.2f}-zero {f[0]:+.3f}({f[1]:.3f})"
    print(s)
b = 39; x = am[:, b]; dev = np.abs(x - np.median(x)); q = np.percentile(dev, [50, 99, 99.9])
print("400 steps: |AD - median| percentiles 50/99/99.9:", np.round(q, 3), " max", round(float(dev.max()), 2))
for k in (5, 10):
    w = dev < k * q[0] * 1.4826
    print(f"  windows with |dev| < {k} x typical width: {w.mean():.3f};  AD {x[w].mean():.3f}  zero {az[w, b].mean():.3f}"
          f"  FD0.08 {fd[w, 1, b].mean():.3f}")
