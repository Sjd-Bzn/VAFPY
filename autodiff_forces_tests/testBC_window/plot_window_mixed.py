"""Window-length figure of tests B/C with the HF value and the mixed estimator of <H1> (run_mixed.py).
usage: plot_window_mixed.py WIN_DIR MIXED_DIR OUT.pdf"""
import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

win_dir, mix_dir, out = sys.argv[1:4]
REF, REF_ERR = 104.30, 0.63            # independent fixed-lambda runs (lambda = +-0.05), test C


def blocked(x, nchain, nb=10):
    """mean and error from blocks of consecutive samples per chain; x has shape (nchain, nsamples, ...)"""
    bm = np.concatenate([np.array([b.mean(axis=0) for b in np.array_split(c, nb)]) for c in x])
    return bm.mean(axis=0), bm.std(axis=0, ddof=1) / np.sqrt(len(bm))


W = [np.load(f) for f in sorted(glob.glob(f"{win_dir}/win_*.npz"))]
block = int(W[0]["block"])
steps = block * (np.arange(W[0]["dE_martin"].shape[1]) + 1)
M = [np.load(f) for f in sorted(glob.glob(f"{mix_dir}/mixed_*.npz"))]
n = min(len(m["V"]) for m in M)
v_mix, e_mix = blocked(np.array([m["V"][:n] for m in M]), len(M), nb=20)
v_hf = float(M[0]["v_hf"])
print(f"HF <H1> {v_hf:.4f}   mixed <H1> {v_mix:.4f}({e_mix:.4f})  [{len(M)} chains x {n} blocks]")

fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.axhspan(REF - REF_ERR, REF + REF_ERR, color="0.88", lw=0, label=r"independent runs at $\lambda=\pm0.05$")
ax.axhline(v_hf, color="0.35", ls="--", lw=1.2, label=f"HF: {v_hf:.2f} eV")
ax.axhline(v_mix, color="#8a5cc2", ls=":", lw=1.6, label=f"mixed estimator: {v_mix:.3f}({1000*e_mix:.0f}) eV")
for key, lab, col, mk in (("dE_martin", "derivative, Martin's rule", "#2a78d6", "o"),
                          ("dE_zero", "derivative, zero rule", "#eb6834", "s"),
                          ("dE_off", "derivative, no population control", "#1baf7a", "^")):
    m, e = blocked(np.array([z[key] for z in W]), len(W))
    sel = np.r_[0:10, 11:40:3, 41:100:6]
    ax.errorbar(np.r_[0, steps[sel]], np.r_[v_mix, m[sel]], yerr=np.r_[e_mix, e[sel]], color=col, marker=mk, ms=3.5,
                lw=1.2, capsize=2, label=lab)
    if key == "dE_martin":
        print("Martin at 10/40/100/200/400/1000 steps:", " ".join(f"{m[i]:.3f}({e[i]:.3f})" for i in (0, 3, 9, 19, 39, 99)))
ax.plot([0], [v_mix], "o", color="#8a5cc2", ms=6, zorder=5)
ax.annotate("0 steps:\nmixed", (0, v_mix), xytext=(60, v_mix - 0.42), fontsize=8, color="#8a5cc2",
            arrowprops=dict(arrowstyle="-", color="#8a5cc2", lw=0.8))
ax.set_xlim(-20, 1020)
ax.set_ylim(v_mix - 0.6, 105.1)
ax.set_xlabel(r"steps in the window ($\Delta\tau=2.5\times10^{-4}$ eV$^{-1}$)")
ax.set_ylabel(r"$\langle \mathrm{d}E/\mathrm{d}\lambda\rangle$ (eV)")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=7.5, loc="lower right", frameon=True)
fig.tight_layout()
fig.savefig(out)
print("saved", out)
