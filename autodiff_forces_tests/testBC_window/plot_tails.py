"""Distribution of the per-window derivatives (data of run_eq_window.py): are there heavy tails (infinite variance)?
Rows: E1, E2. Columns: (1) histogram at the window end, (2) tail plot: fraction of windows with |x - median| > t
on log-log axes, against a Gaussian of the same robust width and the power laws t^-1 and t^-2 (tail exponent 2 is the
border of infinite variance), (3) running mean and naive error bar as windows are added.
usage: plot_tails.py DIR OUT.pdf [STEPS]"""
import sys, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z = [np.load(f) for f in sorted(glob.glob(sys.argv[1] + "/eqw*_*.npz"))]
blk = int(Z[0]["block"]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400; i = steps // blk - 1
nwin = min(len(z["E"]) for z in Z)
# windows in time order across chains: chain c contributes its windows in order; interleave so the running mean mixes chains
cat = lambda k: np.stack([z[k][:nwin, i] for z in Z], axis=1).reshape(-1)
SER = (("martin", "#2a78d6", "derivative, Martin's rule"), ("off", "#1baf7a", "derivative, no population control"),
       ("zero", "#eb6834", "derivative, zero rule"))


def hill(x, k):
    """Hill estimate of the tail exponent alpha (P(|x|>t) ~ t^-alpha) from the k largest deviations."""
    a = np.sort(np.abs(x - np.median(x)))[::-1]
    return k / np.sum(np.log(a[:k] / a[k]))


def describe(x):
    med = np.median(x); rs = 1.4826 * np.median(np.abs(x - med)); s = x.std(ddof=1)
    return dict(mean=x.mean(), se=s / np.sqrt(len(x)), std=s, rstd=rs, kurt=stats.kurtosis(x),
                zmax=np.abs(x - x.mean()).max() / s, a24=hill(x, 24), a48=hill(x, 48),
                tmean=stats.trim_mean(x, 0.05), med=med)


fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
report = []
for r, p in enumerate(("E1", "E2")):
    norm = cat(p)
    a, b, c = ax[r]
    # (1) histogram
    for mode, col, lab in SER[:1]:
        x = cat(f"{p}d_{mode}")
        a.hist(x, bins=60, color=col, alpha=0.75, label=lab)
    a.hist(norm, bins=30, color="0.2", alpha=0.6, label="normal AFQMC")
    a.set_yscale("log"); a.set_xlabel(f"{p} per window (eV)"); a.set_ylabel("windows")
    a.set_title(f"({'ad'[r]}) {p}: values of the {len(norm)} windows at {steps} steps", fontsize=10)
    a.legend(fontsize=8)
    # (2) tail plot
    for mode, col, lab in SER:
        x = cat(f"{p}d_{mode}"); d = describe(x)
        t = np.sort(np.abs(x - d["med"]))[::-1] / d["rstd"]
        b.loglog(t, np.arange(1, len(t) + 1) / len(t), color=col, lw=1.4, label=f"{lab} (alpha ~ {d['a48']:.1f})")
        report.append((p, mode, d))
    dn = describe(norm)
    t = np.sort(np.abs(norm - dn["med"]))[::-1] / dn["rstd"]
    b.loglog(t, np.arange(1, len(t) + 1) / len(t), color="0.2", lw=1.4, label=f"normal AFQMC (alpha ~ {dn['a48']:.1f})")
    report.append((p, "normal", dn))
    tt = np.logspace(-1.5, 1.6, 200)
    b.loglog(tt, 2 * stats.norm.sf(tt), color="0.5", ls="--", lw=1, label="Gaussian")
    for al, ls in ((1, ":"), (2, "-.")):
        b.loglog(tt[tt > 2], 0.05 * (tt[tt > 2] / 2) ** -al, color="k", ls=ls, lw=0.9,
                 label=f"power law, alpha = {al}" + (" (variance infinite below)" if al == 2 else ""))
    b.set_ylim(0.5 / len(norm), 1.2); b.set_xlim(0.03, 60)
    b.set_xlabel("|x - median| / robust width"); b.set_ylabel("fraction of windows beyond")
    b.set_title(f"({'be'[r]}) {p}: tail of the distribution", fontsize=10)
    b.legend(fontsize=7.2, loc="lower left"); b.grid(alpha=0.25, which="both", lw=0.4)
    # (3) running mean
    n = np.arange(1, len(norm) + 1)
    for mode, col, lab in SER[:2]:
        x = cat(f"{p}d_{mode}")
        rm = np.cumsum(x) / n
        rse = np.array([x[:k].std(ddof=1) / np.sqrt(k) if k > 1 else np.nan for k in n])
        c.plot(n, rm, color=col, lw=1.3, label=lab)
        c.fill_between(n, rm - rse, rm + rse, color=col, alpha=0.18)
    c.set_xlabel("number of windows used"); c.set_ylabel(f"running mean of {p} (eV)")
    c.set_title(f"({'cf'[r]}) {p}: running mean and naive error bar", fontsize=10)
    c.set_ylim(*((103.4, 105.6) if p == "E1" else (1.5, 5.5)))
    c.legend(fontsize=8); c.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(sys.argv[2]); fig.savefig(sys.argv[2].rsplit(".", 1)[0] + ".png", dpi=150)

print(f"{len(Z)} chains x {nwin} windows, values at {steps} steps")
print("quantity  rule     mean     naive err  5%-trimmed  median   std    robust std  std/robust  kurtosis  max|z|  alpha(k=24) alpha(k=48)")
for p, mode, d in report:
    print(f"{p:3s}  {mode:7s} {d['mean']:9.3f} {d['se']:9.3f} {d['tmean']:10.3f} {d['med']:9.3f} {d['std']:7.3f} {d['rstd']:9.3f}"
          f" {d['std'] / d['rstd']:9.2f} {d['kurt']:9.1f} {d['zmax']:7.1f} {d['a24']:9.1f} {d['a48']:10.1f}")
