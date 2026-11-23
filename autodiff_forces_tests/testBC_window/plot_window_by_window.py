"""Window-length test resolved by window number: the climb of dE/dlam within a window is the same in every window
after the equilibration, so it is not an equilibration of the walkers. Martin's rule, first N windows of every chain.
With E1E2_DIR (output of run_e1e2_window.py) the E1 of normal AFQMC (weighted average of the one-body energy of the
walkers, as printed by measure_energy) is added along the window and against the window number.
usage: plot_window_by_window.py WIN_DIR OUT.pdf [N_WINDOWS] [E1E2_DIR]"""
import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

win_dir, out = sys.argv[1:3]
nwin = int(sys.argv[3]) if len(sys.argv) > 3 else 50
e1e2_dir = sys.argv[4] if len(sys.argv) > 4 else None
E1_HF = 101.3575976                     # VASP E1 = <H1> of the HF trial

W = [np.load(f) for f in sorted(glob.glob(f"{win_dir}/win_*.npz"))]
block = int(W[0]["block"])
nblk = W[0]["dE_martin"].shape[1]
steps = block * (np.arange(nblk) + 1)
D = np.array([z["dE_martin"].real for z in W])[:, :nwin]      # chain, window, block
E = np.array([z["E"].real for z in W])[:, :nwin].mean(axis=2)  # chain, window
nch = len(W)
N1 = None
if e1e2_dir:                              # normal AFQMC E1 after every block, chain, window, block
    Z = [np.load(f) for f in sorted(glob.glob(f"{e1e2_dir}/e1e2w_*.npz"))]
    nz = min(len(z["E1_walkers"]) for z in Z)
    N1 = np.array([z["E1_walkers"][:nz].real for z in Z])[:, :nwin]
    steps1 = int(Z[0]["block"]) * (np.arange(N1.shape[2]) + 1)


def mean_err(x):
    x = x.reshape(-1, *x.shape[2:]) if x.ndim > 1 else x
    return x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(len(x))


fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))

# (a) dE/dlam against the steps in the window, groups of 10 windows
ng = nwin // 10
blues = plt.cm.Blues(np.linspace(0.45, 1.0, ng))
for g in range(ng):
    m, e = mean_err(D[:, 10 * g:10 * g + 10])
    ax[0].fill_between(steps, m - e, m + e, color=blues[g], alpha=0.25, lw=0)
    ax[0].plot(steps, m, color=blues[g], lw=1.3, label=f"windows {10 * g + 1}-{10 * g + 10}")
ax[0].axhline(E1_HF, color="0.35", ls="--", lw=1.1, label=f"HF (VASP $E_1$) {E1_HF:.2f}")
if N1 is not None:
    m, e = mean_err(N1)
    ax[0].fill_between(steps1, m - e, m + e, color="#8a5cc2", alpha=0.25, lw=0)
    ax[0].plot(steps1, m, color="#8a5cc2", lw=1.4, ls="-.", label=f"normal AFQMC $E_1$ {N1.mean():.2f}")
m_all, e_all = mean_err(D)
for s_ in (10, 40, 100, 400):
    i = s_ // block - 1
    ax[0].plot(steps[i], m_all[i], "o", color="k", ms=3.5, zorder=5)
    ax[0].annotate(f"{m_all[i]:.2f}", (steps[i], m_all[i]), xytext=(4, -11) if s_ < 400 else (-8, 9), textcoords="offset points", fontsize=7.5)
ax[0].set_xscale("log")
ax[0].set_ylim(101.0, 105.4)
ax[0].set_xlabel(r"steps in the window ($\Delta\tau=2.5\times10^{-4}$ eV$^{-1}$)")
ax[0].set_ylabel(r"$\langle \mathrm{d}E/\mathrm{d}\lambda\rangle$ (eV)")
ax[0].set_title(f"(a) every group of windows climbs the same way", fontsize=10)
ax[0].legend(fontsize=7.5, loc="upper left", frameon=True)

# (b) value at 10, 40, 400 steps against the window number, bins of 5 windows
bw = 5
xb = np.arange(nwin // bw) * bw + (bw + 1) / 2
for s, col, mk in ((10, "#eb6834", "s"), (40, "#1baf7a", "^"), (400, "#2a78d6", "o")):
    i = s // block - 1
    me = [mean_err(D[:, k * bw:(k + 1) * bw, i]) for k in range(nwin // bw)]
    ax[1].errorbar(xb, [a for a, _ in me], yerr=[b for _, b in me], color=col, marker=mk, ms=4, lw=1.1, capsize=2,
                   label=f"after {s} steps")
ax[1].axhline(E1_HF, color="0.35", ls="--", lw=1.1)
if N1 is not None:
    nb1 = N1.shape[1] // bw
    me = [mean_err(N1[:, k * bw:(k + 1) * bw].mean(axis=2)) for k in range(nb1)]
    ax[1].errorbar(np.arange(nb1) * bw + (bw + 1) / 2, [a for a, _ in me], yerr=[b for _, b in me], color="#8a5cc2",
                   marker="D", ms=3.5, lw=1.1, ls="-.", capsize=2, label="normal AFQMC $E_1$")
ax[1].set_ylim(101.0, 105.4)
ax[1].set_xlabel(f"window number (each window {steps[-1]} steps)")
ax[1].set_ylabel(r"$\langle \mathrm{d}E/\mathrm{d}\lambda\rangle$ (eV)")
ax[1].set_title("(b) no trend with the window number", fontsize=10)
ax[1].legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.61), frameon=True)

# (c) energy against the window number
me = [mean_err(E[:, k * bw:(k + 1) * bw]) for k in range(nwin // bw)]
ax[2].errorbar(xb, [a for a, _ in me], yerr=[b for _, b in me], color="0.2", marker="o", ms=4, lw=1.1, capsize=2)
ax[2].axhline(E.mean(), color="0.6", lw=0.8)
ax[2].set_ylim(E.mean() - 0.3, E.mean() + 0.3)
ax[2].set_xlabel(f"window number (each window {steps[-1]} steps)")
ax[2].set_ylabel("AFQMC energy (eV)")
ax[2].set_title("(c) the walkers are equilibrated", fontsize=10)

for a in ax:
    a.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(out)
fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=150)
print(f"{nch} chains x {nwin} windows; energy {E.mean():.3f}; saved {out}")
