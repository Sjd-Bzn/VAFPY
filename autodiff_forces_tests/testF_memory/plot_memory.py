"""Memory of the derivative against window length (VSC5 CPU, one core, 100 walkers, double precision)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

R = [json.loads(l) for l in open(sys.argv[1])]
out = sys.argv[2]
modes = [("plain", "energy only", "#52514e"), ("fwd", "forward mode", "#2a78d6"),
         ("rev", "reverse mode", "#eb6834"), ("ckpt", "reverse + checkpointing", "#1baf7a")]
systems = [("gamma", r"diamond, $\Gamma$ (16 orbitals)"), ("k222", r"diamond, $2\times2\times2$ $k$-mesh")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
for ax, (sysname, title) in zip(axes, systems):
    for mode, label, col in modes:
        pts = sorted((r["L"], (r["rss_peak_bytes"] - r["rss_base_bytes"]) / 1e9) for r in R
                     if r["system"] == sysname and r["mode"] == mode and r["nw"] == 100 and r["prec"] == "Double")
        x, y = zip(*pts)
        ax.plot(x, y, "-o", color=col, lw=2, ms=4, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ticks = [0.2, 0.5, 1, 2, 5] if sysname == "gamma" else [0.5, 1, 2, 5, 10, 20, 50]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xlabel("window length (steps)")
    ax.set_title(title, fontsize=9, color="#0b0b0b")
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.tick_params(colors="#52514e")
axes[0].set_ylabel("memory (GB)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig(out)
