"""Finite differences with the same random numbers against the forward-mode derivative (data of run_fd_check.py):
error vs step h, all windows and window lengths. (a) median and largest error over the smooth cases, (b) fraction of
cases with a jump (|fd - ad| > 1e-3 eV: a comb decision or a kink changed between -h and +h).
usage: plot_fd_check.py DIR OUT.pdf"""
import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = [np.load(f) for f in sorted(glob.glob(sys.argv[1] + "/fd_*.npz"))]
H = F[0]["h"]; nwin = sum(len(f["E1_off_ad"]) for f in F); nb = F[0]["E1_off_ad"].shape[1]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
STY = {("E1", "off"): ("#1baf7a", "o", "-"), ("E2", "off"): ("#1baf7a", "s", "--"),
       ("E1", "zero"): ("#eb6834", "o", "-"), ("E2", "zero"): ("#eb6834", "s", "--")}
print(f"{nwin} windows x {nb} window lengths")
for (p, m), (col, mk, ls) in STY.items():
    ad = np.concatenate([f[f"{p}_{m}_ad"] for f in F])
    med, mx, jump = [], [], []
    for h in H:
        err = np.abs(np.concatenate([f[f"{p}_{m}_fd{h:g}"] for f in F]) - ad)
        ok = err < 1e-3
        med.append(np.median(err)); mx.append(err[ok].max()); jump.append(100 * (~ok).mean())
    lab = f"{p}, " + ("no population control" if m == "off" else "zero rule (comb)")
    ax[0].loglog(H, med, color=col, marker=mk, ls=ls, label=lab + ", median")
    ax[0].loglog(H, mx, color=col, marker=mk, ls=ls, alpha=0.4, mfc="none", label=lab + ", largest (no jump)")
    ax[1].semilogx(H, jump, color=col, marker=mk, ls=ls, label=lab)
    print(f"{p} {m:4s} " + "  ".join(f"h={h:g}: median {a:.0e} largest {b:.0e} jumps {c:.1f}%" for h, a, b, c in zip(H, med, mx, jump)))
ax[0].set_xlabel(r"finite-difference step $h$"); ax[0].set_ylabel(r"$|\mathrm{FD} - \mathrm{AD}|$ (eV)")
ax[0].set_title(f"(a) central difference vs AD, same random numbers ({nwin} windows x {nb} lengths)", fontsize=9.5)
ax[0].legend(fontsize=6.8, ncol=1); ax[0].grid(alpha=0.25, which="both", lw=0.4)
ax[1].set_xlabel(r"finite-difference step $h$"); ax[1].set_ylabel("cases with a jump (%)")
ax[1].set_title(r"(b) jumps: $|\mathrm{FD}-\mathrm{AD}|>10^{-3}$ eV (comb decision flipped)", fontsize=9.5)
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25, which="both", lw=0.4)
fig.tight_layout(); fig.savefig(sys.argv[2]); fig.savefig(sys.argv[2].rsplit(".", 1)[0] + ".png", dpi=150)
