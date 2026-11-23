"""E1, E2 and E1+E2 along the window: normal AFQMC (weighted average of e1, eh+ex of the walkers) and as derivatives, same trajectory.
usage: plot_e1e2_window.py DIR OUT.pdf"""
import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = [np.load(f) for f in sorted(glob.glob(sys.argv[1] + "/e1e2w_*.npz"))]
blk = int(Z[0]["block"])
c = lambda k: np.concatenate([z[k] for z in Z])                     # window, block
steps = blk * (np.arange(c("E").shape[1]) + 1)
me = lambda x: (x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x)))
HF1, HF2 = 101.3575976, 12.2774798
E = c("E")
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
for a, (k, hf, title) in zip(ax, (("1", HF1, "one-body part E1"), ("2", HF2, "two-body part E2"), (None, HF1 + HF2, "E1 + E2"))):
    if k:
        w, dv = c(f"E{k}_walkers"), c(f"E{k}_deriv")
    else:
        w, dv = c("E1_walkers") + c("E2_walkers"), c("E1_deriv") + c("E2_deriv")
        m, e = me(E); a.plot(steps, m, color="0.2", lw=1.0, ls=":", label=f"AFQMC energy {E.mean():.2f}")
    for x, col, mk, lab in ((w, "#eb6834", "s", "normal AFQMC (as measure_energy)"),
                            (dv, "#2a78d6", "o", "derivative (Martin's rule)")):
        m, e = me(x)
        a.fill_between(steps, m - e, m + e, color=col, alpha=0.25, lw=0)
        a.plot(steps, m, color=col, lw=1.4, label=lab)
    if k:
        a.axhline(hf, color="0.35", ls="--", lw=1.1, label=f"HF {hf:.2f}")
    a.set_xlabel(r"steps in the window ($\Delta\tau=2.5\times10^{-4}$ eV$^{-1}$)")
    a.set_ylabel("eV"); a.set_title(title, fontsize=10); a.grid(alpha=0.25, lw=0.5)
ax[2].set_ylim(E.mean() - 1.2, E.mean() + 0.8)
for a, loc in zip(ax, ("center right", "center right", "lower center")):
    a.legend(fontsize=7.5, loc=loc, frameon=True)
fig.tight_layout(); fig.savefig(sys.argv[2]); fig.savefig(sys.argv[2].rsplit(".", 1)[0] + ".png", dpi=150)
print(f"{E.shape[0]} windows")
print("steps   E1 walkers  E1 deriv   E2 walkers  E2 deriv   sum walkers  sum deriv   E")
for s in (10, 20, 40, 100, 200, 300, 400):
    i = s // blk - 1
    f = lambda x: "%7.3f(%3.0f)" % (x[:, i].mean(), 1000 * x[:, i].std(ddof=1) / np.sqrt(len(x)))
    print(f"{s:4d}  " + "  ".join(f(x) for x in (c("E1_walkers"), c("E1_deriv"), c("E2_walkers"), c("E2_deriv"),
          c("E1_walkers") + c("E2_walkers"), c("E1_deriv") + c("E2_deriv"), E)))
