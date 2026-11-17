"""dE/dx (eV/A) against window length: derivative (Martin's and the zero rule) and correlated finite difference."""
import glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
pat = sys.argv[1] if len(sys.argv) > 1 else "prod_*.npz"
out = sys.argv[2] if len(sys.argv) > 2 else "force_window.pdf"
Z = [np.load(f) for f in sorted(glob.glob(pat))]
cat = lambda k: np.concatenate([z[k] for z in Z])
hs = Z[0]["hs"]; block = int(Z[0]["block"])
am, az, Ep, Em = cat("ad_martin"), cat("ad_zero"), cat("Ep"), cat("Em")
fd = (Ep - Em) / (2 * hs[None, :, None])
def me(x, nb=80):
    b = np.array([c.mean() for c in np.array_split(x, nb)]); return x.mean(), b.std(ddof=1) / np.sqrt(nb)
steps = (np.arange(am.shape[1]) + 1) * block
sel = [b for b in range(am.shape[1]) if steps[b] in (20, 40, 60, 100, 150, 200, 300, 400)]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.axhline(0.8641, color="#52514e", lw=1.2, ls="--")
ax.text(410, 0.80, "HF", color="#52514e", va="top", ha="right", fontsize=9)
series = [("derivative, Martin's rule", am, "#eb6834", "o", -3), ("derivative, zero rule", az, "#2a78d6", "s", 3)]
for j, (h, c) in enumerate(zip(hs, ("#1baf7a", "#8e5bd0", "#c0a020"))):
    series.append((f"finite difference, h = {h:.2f} Å", fd[:, j], c, "^", 6 * (j - 1)))
for lab, x, c, mk, dx in series:
    m = np.array([me(x[:, b]) for b in sel])
    ax.errorbar(steps[sel] + dx, m[:, 0], m[:, 1], color=c, marker=mk, ms=4.5, lw=1.4, capsize=2, label=lab)
ax.set_xlabel("steps in the window (block = 10 steps)")
ax.set_ylabel("dE/dx  (eV/Å)")
ax.set_xlim(0, 420)
ax.grid(color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(out); print("saved", out, len(Z), "chains", am.shape[0], "windows")
