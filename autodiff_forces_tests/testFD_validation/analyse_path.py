"""Pathwise check: FD along the same trajectory against the forward-mode derivative.
off : FD(h) vs AD(off)                -> must agree for all windows (smooth trajectory)
comb: FD(h) vs AD(zero) and AD(Martin) -> FD agrees with the zero rule except in windows with a comb jump"""
import glob
import numpy as np
Z = [np.load(f) for f in sorted(glob.glob("path_*.npz"))]
hs = Z[0]["hs"]
cat = lambda k: np.concatenate([z[k] for z in Z])
aoff, az, am, foff, fcomb = cat("ad_off"), cat("ad_zero"), cat("ad_martin"), cat("fd_off"), cat("fd_comb")
N = aoff.shape[0]
print(f"{N} windows of {aoff.shape[1]} blocks")
for L in (40, 100, 200, 400):
    b = L // 10 - 1
    print(f"L = {L}:  mean AD off {aoff[:, b].mean():.4f}  zero {az[:, b].mean():.4f}  Martin {am[:, b].mean():.4f}")
    for j, h in enumerate(hs):
        doff = np.abs(foff[:, j, b] - aoff[:, b])
        dz = np.abs(fcomb[:, j, b] - az[:, b])
        dm = np.abs(fcomb[:, j, b] - am[:, b])
        jump = dz > 1e-2
        print(f"   h={h:7.0e}  off: max|FD-AD| {doff.max():.1e}   comb: median|FD-zero| {np.median(dz):.1e}"
              f"  median|FD-Martin| {np.median(dm):.1e}  windows with jump {100*jump.mean():5.1f}%")
