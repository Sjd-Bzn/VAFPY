"""Figure: equilibration from HF (left column) and the window after equilibration (right column) for E1, E2 and E.
Normal AFQMC (trial on the left) and the derivative with Martin's rule, the zero rule and without population control,
with HF and FCI (HF on the left, Psi0 on both sides). Data from run_eq_window.py.
usage: plot_eq_window.py DIR OUT.pdf [REVDIR]   (REVDIR: run_reverse_check.py output, reverse mode added)"""
import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = [np.load(f) for f in sorted(glob.glob(sys.argv[1] + "/eqw_*.npz"))]
blk = int(Z[0]["block"])
nwin = min(len(z["E"]) for z in Z)
cat = lambda k: np.concatenate([z[k][:nwin] for z in Z])                 # (windows, blocks)
eqm = lambda k: np.mean([z[f"eq_{k}"] for z in Z], axis=0)
me = lambda x: (x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x)))

HF = dict(E1=101.3575976, E2=12.2774798, E=113.6350774)
MIX = dict(E1=101.223633, E2=6.694595, E=107.918228)                 # FCI, HF on the left
PURE = dict(E1=104.982744, E2=2.935484, E=107.918228)                # FCI, Psi0 on both sides
SLOPE = dict(E1=(104.54, 0.11), E2=(3.24, 0.10))     # slope of the AFQMC energy, lambda scans (H1 or H2 scaled)
REV = None
if len(sys.argv) > 3:                         # reverse mode (plain, checkpointed) on the same windows, Martin's rule
    R = [np.load(f) for f in sorted(glob.glob(sys.argv[3] + "/rev_*.npz"))]
    nr = min(len(r["E1_rev"]) for r in R)
    rc = lambda k: np.concatenate([r[k][:nr] for r in R])
    REV = dict(sel=R[0]["sel"], n=nr, **{k: rc(k) for k in ("E1_rev", "E1_revck", "E2_rev", "E2_revck", "E1_fwd", "E2_fwd")})
    REV["E_rev"] = REV["E1_rev"] + REV["E2_rev"]; REV["E_revck"] = REV["E1_revck"] + REV["E2_revck"]
SLOPE["E"] = (SLOPE["E1"][0] + SLOPE["E2"][0], float(np.hypot(SLOPE["E1"][1], SLOPE["E2"][1])))   # sum of the two

C_NORM, C_HF, C_FCI = "0.15", "0.45", "#8a5cc2"


def fci_lines(a, k):
    """FCI with HF on the left (dotted) and with Psi0 on both sides (solid); for E both are E0 and coincide."""
    same = " (both = $E_0$)" if k == "E" else ""
    a.axhline(PURE[k], color=C_FCI, ls="-", lw=1.2, label=fr"FCI, $\Psi_0$ on both sides {PURE[k]:.3f}{same}")
    a.axhline(MIX[k], color=C_FCI, ls=":", lw=2.4 if k == "E" else 1.6, label=f"FCI, HF on the left {MIX[k]:.3f}{same}")

MODES = (("martin", "#2a78d6", "o", "derivative, Martin's rule"),
         ("zero", "#eb6834", "s", "derivative, zero rule"),
         ("off", "#1baf7a", "^", "derivative, no population control"))

neq = len(Z[0]["eq_E"])
x_eq = blk * np.arange(1, neq + 1)
x_w = blk * np.arange(1, cat("E").shape[1] + 1)
sel = np.r_[0:10, 11:cat("E").shape[1]:3]
if sel[-1] != len(x_w) - 1:
    sel = np.r_[sel, len(x_w) - 1]

fig, ax = plt.subplots(3, 2, figsize=(13, 11.5), gridspec_kw=dict(width_ratios=(1, 1.35)))
rows = (("E1", "one-body part $E_1$", (100.9, 105.4), (100.9, 105.4)),
        ("E2", "two-body part $E_2$", (2.4, 12.8), (2.4, 9.7)),
        ("E", "energy $E=E_1+E_2$", (107.3, 114.0), (107.3, 109.7)))
for r, (k, name, ylim_eq, ylim_w) in enumerate(rows):
    a, b = ax[r]
    # ---- equilibration ----
    a.plot(np.r_[0, x_eq], np.r_[HF[k], eqm(k)], color=C_NORM, lw=0.9, label="normal AFQMC (mean of %d chains)" % len(Z))
    a.axhline(HF[k], color=C_HF, ls="--", lw=1.1, label=f"HF {HF[k]:.2f}")
    fci_lines(a, k)
    a.set_ylim(*ylim_eq); a.set_xlim(0, x_eq[-1])
    a.set_title(f"({'ace'[r]}) equilibration from HF: {name}", fontsize=10)
    a.set_ylabel("eV"); a.grid(alpha=0.25, lw=0.5)
    a.set_xlabel(r"steps from the HF start ($\Delta\tau=2.5\times10^{-4}$ eV$^{-1}$)")
    a.legend(fontsize=7.5, loc="upper right" if k != "E1" else "center right", frameon=True)
    # ---- window ----
    if k == "E":
        norm = cat("E")
        der = {m: cat(f"E1d_{m}") + cat(f"E2d_{m}") for m, *_ in MODES}
    else:
        norm = cat(k)
        der = {m: cat(f"{k}d_{m}") for m, *_ in MODES}
    m_, e_ = me(norm)
    b.plot(x_w, m_, color=C_NORM, lw=1.4, label=f"normal AFQMC {norm.mean():.3f}")
    for mode, col, mk, lab in MODES:
        m_, e_ = me(der[mode])
        b.errorbar(x_w[sel], m_[sel], yerr=e_[sel], color=col, marker=mk, ms=3.2, lw=1.2, capsize=2,
                   label=f"{lab}: {m_[-1]:.2f}({100 * e_[-1]:.0f}) at {x_w[-1]} steps")
    if REV is not None:
        xs = x_w[REV["sel"]]
        for kind, mk, ms, lab in (("rev", "D", 9, "reverse mode"), ("revck", "x", 6, "reverse mode, checkpointed")):
            mr, er = me(REV[f"{k}_{kind}"])
            b.plot(xs, mr, ls="none", marker=mk, ms=ms, mfc="none" if mk == "D" else None, mec="#2a78d6",
                   color="#2a78d6", mew=1.1,
                   label=f"{lab}, Martin's rule ({REV['n'] * len(Z)} windows): {mr[-1]:.2f}({100 * er[-1]:.0f})")
    if ylim_w[0] < HF[k] < ylim_w[1]:
        b.axhline(HF[k], color=C_HF, ls="--", lw=1.1, label=f"HF {HF[k]:.2f}")
    fci_lines(b, k)
    if k in SLOPE:
        b.errorbar([x_w[-1] + 18], [SLOPE[k][0]], yerr=[SLOPE[k][1]], color="k", marker="D", ms=5, capsize=3, lw=1.2,
                   label=(fr"sum of the two $\lambda$-scan slopes ($E_1+E_2$, no derivative) " if k == "E" else
                          fr"slope of the AFQMC energy ($\lambda$ scan, no derivative) ") + f"{SLOPE[k][0]:.2f}({100 * SLOPE[k][1]:.0f})")
    b.set_ylim(*ylim_w); b.set_xlim(0, x_w[-1] + 30)
    b.set_title(f"({'bdf'[r]}) window after equilibration: {name}", fontsize=10)
    b.set_ylabel("eV"); b.grid(alpha=0.25, lw=0.5)
    b.set_xlabel(r"steps in the window (derivative set to zero at the start)")
    if k == "E1":
        b.legend(fontsize=7.2, loc="lower right", bbox_to_anchor=(1, 0.03), frameon=True)
    else:
        b.legend(fontsize=7.2, loc="upper right" if k == "E2" else "upper left", frameon=True, ncol=1)
fig.tight_layout()
fig.savefig(sys.argv[2]); fig.savefig(sys.argv[2].rsplit(".", 1)[0] + ".png", dpi=150)

print(f"{len(Z)} chains, {neq * blk} equilibration steps, {norm.shape[0]} windows of {x_w[-1]} steps")
print("equilibration (mean of chains, average of 10 blocks around the step):")
for s in (10, 100, 300, 1000, 2000, neq * blk):
    i = s // blk - 1; j = slice(max(0, i - 5), i + 5)
    print(f"  step {s:5d}: E {eqm('E')[j].mean():8.3f}  E1 {eqm('E1')[j].mean():8.3f}  E2 {eqm('E2')[j].mean():7.3f}")
print("window (mean(error) over windows):")
f = lambda x, i: "%8.3f(%3.0f)" % (x[:, i].mean(), 1000 * x[:, i].std(ddof=1) / np.sqrt(len(x)))
print("  steps  " + "  ".join("%13s" % h for h in ("E1 normal", "E1 Martin", "E1 zero", "E1 off",
                                                       "E2 normal", "E2 Martin", "E2 zero", "E2 off")))
for s in (10, 40, 100, 200, 300, x_w[-1]):
    i = s // blk - 1
    print(f"  {s:5d}  " + "  ".join(f(x, i) for x in (cat("E1"), cat("E1d_martin"), cat("E1d_zero"), cat("E1d_off"),
                                                     cat("E2"), cat("E2d_martin"), cat("E2d_zero"), cat("E2d_off"))))
