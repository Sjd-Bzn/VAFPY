"""Statistics report of an AFQMC energy series.

analyse() returns the numbers (error from the integrated autocorrelation time, equilibration check);
write_figure() draws them (matplotlib is optional):
  1. energy trace with the equilibration region (discarded) and the production region,
  2. mean energy versus the discarded fraction (is the equilibration cut long enough?),
  3. blocking curve (error of the mean versus block size),
  4. autocorrelation function rho(t) of the production region,
  5. running sum tau(t) = 1/2 + sum rho and the window where it is cut,
  6. the numbers.
"""
import numpy as np


def autocorrelation(x):
    """Normalised autocorrelation function of x (mean removed), lags 0..N-1."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    f = np.fft.rfft(x - x.mean(), 2 * n)
    rho = np.fft.irfft(f * np.conj(f))[:n]
    return rho / rho[0] if rho[0] != 0.0 else rho


def autocorr_error(x, c=10.0):
    """Same estimator as functions.autocorr_error: returns (error, tau_int, window_closed, window, running_tau, rho)."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), False, 0, np.array([]), np.array([])
    var = np.var(x, ddof=1)
    if var == 0.0:
        return 0.0, 0.5, True, 0, np.array([]), np.array([1.0])
    rho = autocorrelation(x)
    tau, closed, window = 0.5, False, n - 1
    running = np.empty(n - 1)
    for k in range(1, n):
        tau += rho[k]
        running[k - 1] = tau
        if k >= c * tau:
            closed = k <= n // 10
            window = k
            break
    tau = max(tau, 0.5)
    return float(np.sqrt(2.0 * tau * var / n)), float(tau), closed, window, running[:window], rho


def blocking_curve(x, divisor=15, npoints=60):
    """Error of the mean from blocks of increasing size; sizes up to N/divisor."""
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    top = max(2, int(n / divisor))
    sizes = np.unique(np.round(np.logspace(0, np.log10(top), npoints)).astype(int))
    err = []
    for b in sizes:
        nb = n // b
        m = x[:nb * b].reshape(nb, b).mean(1)
        err.append(np.sqrt(np.var(m) / (nb - 1)) if nb > 1 else np.nan)
    return sizes, np.array(err)


def analyse(time, energy, eq_fraction, c=10.0):
    """Numbers of the report. time and energy are the first two columns of the output file."""
    time = np.asarray(time, dtype=float).ravel()
    energy = np.asarray(energy, dtype=float).ravel()
    n = len(energy)
    st = int(n * eq_fraction)
    x = energy[st:]
    err, tau, closed, window, running, rho = autocorr_error(x, c)
    half = len(x) // 2
    m1, m2 = x[:half].mean(), x[half:].mean()
    # each half has an error of about sqrt(2)*err, so the difference of the halves has about 2*err
    drift_z = (m2 - m1) / (2.0 * err) if err > 0 else 0.0
    cuts = np.arange(0.0, 0.5001, 0.025)
    scan = []
    for f in cuts:
        s = energy[int(n * f):]
        scan.append((f, s.mean(), autocorr_error(s, c)[0]))
    return dict(n=n, start=st, time=time, energy=energy, eq_fraction=eq_fraction, mean=float(x.mean()), error=err, tau=tau,
                window_closed=closed, window=window, running_tau=running, rho=rho, n_eff=len(x) / (2.0 * tau),
                time_eq_end=float(time[st]) if st < n else float(time[-1]), time_end=float(time[-1]),
                mean_first_half=float(m1), mean_second_half=float(m2), drift_z=float(drift_z), scan=np.array(scan),
                block=blocking_curve(x))


def summary_lines(r):
    """Text lines for outcar.txt / the screen."""
    return [f"Equilibration region (discarded): samples 0-{r['start']} of {r['n']}, time 0 - {r['time_eq_end']:.6g}",
            f"Production region: samples {r['start']}-{r['n']}, time {r['time_eq_end']:.6g} - {r['time_end']:.6g}",
            f"Mean of the first / second half of the production region = {r['mean_first_half']:.6f} / {r['mean_second_half']:.6f}"
            f" (difference {r['mean_second_half'] - r['mean_first_half']:+.6f}, {r['drift_z']:+.2f} sigma)",
            f"Effective number of independent samples = {r['n_eff']:.0f}"]


def write_figure(r, filename="stat_analysis.png", title=""):
    """Draw the report. Returns True if the figure was written (False when matplotlib is not available)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    t, e, st, n = r["time"], r["energy"], r["start"], r["n"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    # 1: trace
    a = ax[0, 0]
    a.plot(t, e, color="lightgray", lw=0.5, label="every sample")
    nb = max(1, n // 200)
    m = e[:(n // nb) * nb].reshape(-1, nb).mean(1)
    a.plot(t[:(n // nb) * nb].reshape(-1, nb).mean(1), m, color="tab:blue", lw=1.2, label=f"mean of {nb} samples")
    if st > 0:
        a.axvspan(t[0], t[st - 1], color="tab:red", alpha=0.15, label="equilibration (discarded)")
    a.axhspan(r["mean"] - r["error"], r["mean"] + r["error"], xmin=0, xmax=1, color="tab:green", alpha=0.3)
    a.axhline(r["mean"], color="tab:green", lw=1.2, label=f"production mean {r['mean']:.5f}")
    lo, hi = np.percentile(e[st:], [0.5, 99.5]) if n - st > 10 else (e.min(), e.max())
    pad = 4 * (hi - lo) / 2 + 1e-12
    a.set_ylim(r["mean"] - pad, r["mean"] + pad)
    a.set_xlabel("imaginary time"); a.set_ylabel("energy"); a.set_title("energy trace and equilibration region"); a.legend(fontsize=7.5)
    # 2: mean versus cut
    a = ax[0, 1]
    f, mu, er = r["scan"][:, 0], r["scan"][:, 1], r["scan"][:, 2]
    a.errorbar(f, mu, yerr=er, fmt="o-", ms=3, color="tab:blue", capsize=2)
    a.axvline(r["eq_fraction"], color="tab:red", ls="--", label=f"used: {r['eq_fraction']:.3g}")
    a.set_xlabel("discarded fraction of the series"); a.set_ylabel("mean energy"); a.set_title("mean energy vs equilibration cut"); a.legend(fontsize=8)
    # 3: blocking
    a = ax[0, 2]
    s, be = r["block"]
    a.semilogx(s, be, "o-", ms=3, color="tab:blue", label="error from blocks of this size")
    a.axhline(r["error"], color="tab:green", lw=2, label=f"reported error {r['error']:.3g}")
    a.set_xlabel("block size (samples)"); a.set_ylabel("error of the mean"); a.set_title("blocking curve (flat part = error)"); a.legend(fontsize=8); a.grid(alpha=0.3)
    # 4: rho
    a = ax[1, 0]
    L = int(min(len(r["rho"]) - 1, max(10, 2.2 * r["window"])))
    a.plot(np.arange(L + 1), r["rho"][:L + 1], color="tab:blue"); a.axhline(0, color="k", lw=0.6)
    a.axvline(r["window"], color="tab:green", ls="--", label=f"window closes at t = {r['window']}")
    a.set_xlabel("lag t (samples)"); a.set_ylabel("autocorrelation rho(t)"); a.set_title("autocorrelation function"); a.legend(fontsize=8); a.grid(alpha=0.3)
    # 5: running tau
    a = ax[1, 1]
    a.plot(np.arange(1, len(r["running_tau"]) + 1), r["running_tau"], color="tab:red")
    a.axhline(r["tau"], color="tab:green", lw=1.5, label=f"tau = {r['tau']:.1f} samples")
    a.axvline(r["window"], color="tab:green", ls="--")
    a.set_xlabel("lag t (samples)"); a.set_ylabel("tau(t) = 1/2 + sum rho"); a.set_title("integrated autocorrelation time"); a.legend(fontsize=8); a.grid(alpha=0.3)
    # 6: numbers
    a = ax[1, 2]; a.axis("off")
    lines = [title, "", f"E = {r['mean']:.6f} +- {r['error']:.2g}", f"tau_int = {r['tau']:.1f} samples", ("" if r["window_closed"] else "WARNING: window did not close (series too short)")] + summary_lines(r)
    import textwrap
    text = "\n".join(textwrap.fill(s, width=62, subsequent_indent="    ") if s else "" for s in lines)
    a.text(0.0, 1.0, text, va="top", ha="left", fontsize=9, family="monospace")
    fig.tight_layout(); fig.savefig(filename, dpi=100); plt.close(fig)
    return True
