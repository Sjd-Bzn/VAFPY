"""Figure of the lambda scans (normal AFQMC energies with H1 or H2 scaled, no derivative) and FCI in the same Hamiltonian.
Per scan (one row):
  (left)   E(lam) minus the FCI tangent line at lam = 0, AFQMC and FCI points with the fitted polynomials
  (middle) AFQMC - FCI against lam with a cubic fit (a quadratic is too simple for the H2 scan)
  (right)  slope estimates: two-point central differences against h, polynomial fits, derivative, FCI
AFQMC errors: per run the first N_DROP blocks are dropped, the rest is cut into 10 blocks; error of the mean over
all blocks of the 4 seeds.
usage: plot_lambda_fit.py OUT.pdf SCAN_DIR_H1 FCI_LOG_H1 [SCAN_DIR_H2 FCI_LOG_H2]"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NDROP, NB = 1000, 10
REF005 = "/data/sajad/calcs/autodiff_forces/testBC_window"                   # runs at lam = +-0.05
FCI005 = "/data/sajad/calcs/autodiff_forces/fci/fci_lambda_005.log"
E0_FCI = 107.918228
EXACT = {"h1": 104.982744, "h2": 2.935484}             # <Psi0|H1|Psi0>, <Psi0|H2|Psi0>
# derivatives after 400 steps (run_eq_window.py): Martin's rule, no population control, zero rule
DERIV = {"h1": dict(martin=(104.57, 0.06), off=(104.64, 0.10), zero=(104.28, 0.04)),
         "h2": dict(martin=(3.38, 0.15), off=(3.37, 0.17), zero=(4.20, 0.09))}
DMODES = (("martin", "#eb6834", 0.0, "derivative, Martin's rule"),
          ("off", "#1baf7a", 0.012, "derivative, no population control"),
          ("zero", "0.45", 0.024, "derivative, zero rule (biased)"))
LABEL = {"h1": ("$H_1$", "$E_1$"), "h2": ("$H_2$", "$E_2$")}


def scan(d):
    lams, E, S = [], [], []
    files = glob.glob(f"{d}/scan_lam*_s1.npz")
    for f in sorted(files, key=lambda f: float(re.search(r"scan_lam([-0-9.]+)_s1", f).group(1))):
        lam = float(re.search(r"scan_lam([-0-9.]+)_s1", f).group(1))
        b = []
        for g in sorted(glob.glob(f.replace("_s1.npz", "_s*.npz"))):
            x = np.load(g)["E"].real[NDROP:]
            b += [c.mean() for c in np.array_split(x, NB)]
        lams.append(lam); E.append(np.mean(b)); S.append(np.std(b, ddof=1) / np.sqrt(len(b)))
    return np.array(lams), np.array(E), np.array(S)


def fci(log):
    f = {0.0: E0_FCI}
    for line in open(log):
        a = line.split()
        f[round(float(a[1]), 2)] = float(a[3])
    return f


def polyslope(l, e, s, deg):
    if s is None:
        return np.polyfit(l, e, deg), 0.0
    c, cov = np.polyfit(l, e, deg, w=1 / s, cov="unscaled")
    return c, np.sqrt(cov[-2, -2])


rows = [("h1", sys.argv[2], sys.argv[3])]
if len(sys.argv) > 5:
    rows.append(("h2", sys.argv[4], sys.argv[5]))
out = sys.argv[1]
nrow = len(rows) + (1 if len(rows) == 2 else 0)                # third row: E1 + E2 from the two scans
fig, ax = plt.subplots(nrow, 3, figsize=(15, 4.6 * nrow), squeeze=False)
ST = {}                                                         # slopes per scan, for the E1 + E2 row
C_A, C_F, C_D = "#2a78d6", "#8a5cc2", "#eb6834"
for r, (part, d, log) in enumerate(rows):
    op, en = LABEL[part]
    ST[part] = dict(tp={}, fit={})
    l, E, S = scan(d)
    F = fci(log); lf = np.array(sorted(F)); ef = np.array([F[k] for k in lf])
    tan = lambda x: E0_FCI + EXACT[part] * x                   # FCI tangent at lam = 0
    xx = np.linspace(-0.32, 0.32, 200)
    print(f"=== {op} scaled ({d})")
    print("  lam    AFQMC              FCI          AFQMC - FCI")
    for li, ei, si in zip(l, E, S):
        print(f"  {li:+.2f}  {ei:10.4f}({1e4 * si:3.0f})  {F[round(li, 2)]:11.5f}   {ei - F[round(li, 2)]:+.4f}({1e4 * si:3.0f})")
    fits = {}
    for deg in (3, 4):
        c, e = polyslope(l, E, S, deg); cf, _ = polyslope(lf, ef, None, deg)
        fits[deg] = (c, e, cf)
        print(f"  polynomial degree {deg}: slope AFQMC {c[-2]:.3f}({1e3 * e:.0f})   FCI {cf[-2]:.4f}")
    # ---- (left) E - FCI tangent ----
    a = ax[r, 0]
    c, e, cf = fits[4]
    a.plot(xx, np.polyval(cf, xx) - tan(xx), color=C_F, lw=1.2, label="FCI, polynomial fit (degree 4)")
    a.plot(lf, ef - tan(lf), "s", color=C_F, ms=5, label="FCI energies")
    a.plot(xx, np.polyval(c, xx) - tan(xx), color=C_A, lw=1.2, label="AFQMC, weighted polynomial fit (degree 4)")
    a.errorbar(l, E - tan(l), yerr=S, fmt="o", color=C_A, ms=4, capsize=2, label="AFQMC energies (4 seeds)")
    a.axhline(0, color="0.5", lw=0.8, ls="--")
    a.set_xlabel(fr"$\lambda$ ({op} $\to(1+\lambda)${op})" if part == "h1" else r"$\lambda$ ($L\to(1+\lambda/2)L$)")
    a.set_ylabel(fr"$E(\lambda) - ({E0_FCI:.3f} + {EXACT[part]:.3f}\times\lambda)$ (eV)")
    a.set_title(f"({'ad'[r]}) energy against $\\lambda$, {op} scaled\n(minus the FCI tangent line $E_0^{{\\rm FCI}}+{EXACT[part]:.3f}\\times\\lambda$, $E_0^{{\\rm FCI}}={E0_FCI:.3f}$)", fontsize=10)
    a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=7.5, loc="best")
    # ---- (middle) AFQMC - FCI ----
    a = ax[r, 1]
    diff = E - np.array([F[round(x, 2)] for x in l])
    cd, cov = np.polyfit(l, diff, 3, w=1 / S, cov="unscaled")
    a.errorbar(l, diff, yerr=S, fmt="o", color=C_A, ms=4, capsize=2, label="AFQMC $-$ FCI")
    a.plot(xx, np.polyval(cd, xx), color=C_A, lw=1.0, ls="--",
           label=f"cubic fit: slope at 0 = {cd[-2]:+.2f}({100 * np.sqrt(cov[-2, -2]):.0f}) eV")
    a.axhline(0, color="0.5", lw=0.8)
    a.set_xlabel(r"$\lambda$"); a.set_ylabel("AFQMC $-$ FCI (eV)")
    a.set_title(f"({'be'[r]}) error of the AFQMC energy against $\\lambda$, {op} scaled", fontsize=10)
    a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=7.5, loc="best")
    print(f"  AFQMC - FCI, cubic fit: slope at 0 = {cd[-2]:+.3f}({1e3 * np.sqrt(cov[-2, -2]):.0f})")
    # ---- (right) slope estimates ----
    a = ax[r, 2]
    hs = [h for h in (0.1, 0.2, 0.3) if np.any(np.isclose(l, h)) and np.any(np.isclose(l, -h))]
    for h in hs:
        ip, im = np.argmin(abs(l - h)), np.argmin(abs(l + h))
        cdA = (E[ip] - E[im]) / (2 * h); err = np.hypot(S[ip], S[im]) / (2 * h)
        cdF = (F[round(h, 2)] - F[round(-h, 2)]) / (2 * h)
        a.errorbar([h], [cdA], yerr=[err], fmt="o", color=C_A, ms=5, capsize=2,
                   label="AFQMC, two-point slope $(E(h)-E(-h))/2h$" if h == hs[0] else None)
        a.plot([h], [cdF], "s", color=C_F, ms=5, label="FCI, two-point slope" if h == hs[0] else None)
        print(f"  two-point h = {h}: AFQMC {cdA:.3f}({1e3 * err:.0f})   FCI {cdF:.4f}")
        ST[part]["tp"][h] = (cdA, err, cdF)
    if part == "h1" and os.path.exists(f"{REF005}/ref_lam0.05_s1.npz") and os.path.exists(FCI005):
        # h = 0.05 from the earlier shorter runs (4 seeds x 2000 blocks, first 1000 dropped) and FCI at +-0.05
        def ref(lam):
            b = []
            for s in (1, 2, 3, 4):
                x = np.load(f"{REF005}/ref_lam{lam}_s{s}.npz")["E"].real[1000:]
                b += [c.mean() for c in np.array_split(x, NB)]
            return np.mean(b), np.std(b, ddof=1) / np.sqrt(len(b))
        (ep, sp), (em, sm) = ref("0.05"), ref("-0.05")
        f5 = fci(FCI005)
        cdA, err, cdF = (ep - em) / 0.1, np.hypot(sp, sm) / 0.1, (f5[0.05] - f5[-0.05]) / 0.1
        a.errorbar([0.05], [cdA], yerr=[err], fmt="o", color=C_A, mfc="white", ms=5, capsize=2,
                   label="AFQMC, two-point slope $h=0.05$ (earlier, 5$\\times$ shorter runs)")
        a.plot([0.05], [cdF], "s", color=C_F, ms=5)
        print(f"  two-point h = 0.05: AFQMC {cdA:.3f}({1e3 * err:.0f}) (earlier runs)   FCI {cdF:.4f}")
    for deg, mk, dx in ((3, "^", -0.026), (4, "v", -0.014)):
        c, e, cf = fits[deg]
        ST[part]["fit"][deg] = (c[-2], e, cf[-2])
        a.errorbar([dx], [c[-2]], yerr=[e], fmt=mk, color=C_A, mfc="white", ms=6, capsize=2,
                   label=f"AFQMC, polynomial fit degree {deg}: {c[-2]:.2f}({100 * e:.0f})")
        a.plot([dx], [cf[-2]], mk, color=C_F, mfc="white", ms=6, label=f"FCI, polynomial fit degree {deg}: {cf[-2]:.3f}")
    for mode, col, dx, lab in DMODES:
        v, e = DERIV[part][mode]
        a.errorbar([dx], [v], yerr=[e], fmt="D", color=col, ms=6, capsize=3, label=f"{lab}: {v:.2f}({100 * e:.0f})")
    a.axhline(EXACT[part], color=C_F, lw=1.2, label=fr"FCI, $\Psi_0$ on both sides {EXACT[part]:.3f}")
    a.set_xlim(-0.045, 0.34); a.set_xlabel("$h$ (0: fit or derivative)")
    a.set_ylabel(f"slope at $\\lambda=0$ = {en} (eV)")
    a.set_title(f"({'cf'[r]}) estimates of the slope {en}", fontsize=10)
    a.grid(alpha=0.25, lw=0.5); a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
if nrow == 3:
    # ---- E1 + E2: sums of the slopes of the two scans (sum rule), no scan of the whole H ----
    ax[2, 1].axis("off")
    ax[2, 0].axis("off")
    ax[2, 0].text(0.02, 0.95,
                  "(g) $E_1+E_2$ from the two scans\n\n"
                  "No run perturbs $H_1$ and $H_2$ together.\n"
                  "Each point in (g) is the sum of the two separate\n"
                  "slopes (top row + middle row), like the derivative\n"
                  "sums in the window figure (panel f).\n\n"
                  "Sum rule: scaling all of $H$ by $(1+\\lambda)$ scales the\n"
                  "energy by $(1+\\lambda)$, so the exact $E_1+E_2$ equals\n"
                  "the energy $E$ itself.",
                  va="top", fontsize=9.5, transform=ax[2, 0].transAxes)
    a = ax[2, 2]
    T1, T2 = ST["h1"], ST["h2"]
    print("=== E1 + E2 (sum of the two scans)")
    for i, h in enumerate(sorted(set(T1["tp"]) & set(T2["tp"]))):
        A1, e1, F1 = T1["tp"][h]; A2, e2, F2 = T2["tp"][h]
        a.errorbar([h], [A1 + A2], yerr=[np.hypot(e1, e2)], fmt="o", color=C_A, ms=5, capsize=2,
                   label="AFQMC, sum of two-point slopes" if i == 0 else None)
        a.plot([h], [F1 + F2], "s", color=C_F, ms=5, label="FCI, sum of two-point slopes" if i == 0 else None)
        print(f"  two-point h = {h}: AFQMC {A1 + A2:.3f}({1e3 * np.hypot(e1, e2):.0f})   FCI {F1 + F2:.4f}")
    for deg, mk, dx in ((3, "^", -0.026), (4, "v", -0.014)):
        b1, e1, f1 = T1["fit"][deg]; b2, e2, f2 = T2["fit"][deg]
        a.errorbar([dx], [b1 + b2], yerr=[np.hypot(e1, e2)], fmt=mk, color=C_A, mfc="white", ms=6, capsize=2,
                   label=f"AFQMC, sum of fit slopes, degree {deg}: {b1 + b2:.2f}({100 * np.hypot(e1, e2):.0f})")
        a.plot([dx], [f1 + f2], mk, color=C_F, mfc="white", ms=6, label=f"FCI, sum of fit slopes, degree {deg}: {f1 + f2:.3f}")
        print(f"  fit degree {deg}: AFQMC {b1 + b2:.3f}({1e3 * np.hypot(e1, e2):.0f})   FCI {f1 + f2:.4f}")
    for (v, e), col, dx, lab in (((107.95, 0.11), "#eb6834", 0.0, "derivative sum, Martin's rule"),
                                  ((108.00, 0.08), "#1baf7a", 0.012, "derivative sum, no population control"),
                                  ((108.47, 0.05), "0.45", 0.024, "derivative sum, zero rule (biased)")):
        a.errorbar([dx], [v], yerr=[e], fmt="D", color=col, ms=6, capsize=3, label=f"{lab}: {v:.2f}({100 * e:.0f})")
    a.axhline(108.01, color="0.15", lw=1.2, ls="-.", label="AFQMC energy $E$ (normal AFQMC) 108.01")
    a.axhline(E0_FCI, color=C_F, lw=1.2, label=f"FCI energy $E_0$ {E0_FCI:.3f} (= exact $E_1+E_2$)")
    a.set_xlim(-0.045, 0.34); a.set_xlabel("$h$ (0: fit or derivative)")
    a.set_ylabel("$E_1+E_2$ (eV)")
    a.set_title("(g) estimates of $E_1+E_2$ (sum of the two scans)", fontsize=10)
    a.grid(alpha=0.25, lw=0.5)
    hl = a.get_legend_handles_labels()
    ax[2, 1].legend(*hl, fontsize=7.5, loc="center", frameon=True, title="(g)", title_fontsize=8)
fig.tight_layout()
fig.savefig(out); fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=150)
