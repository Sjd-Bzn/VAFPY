"""Timers, operation counts and the run environment for the timing report of vafpy (timing.txt).

TIMING in vafpy.in:
  0  no timers, no report
  1  (default) the parts of the run: set-up, propagation, energy, re-orthogonalisation, population control, output,
     analysis; with JAX the device is synchronised once at the end of each part (the step already waits for the device
     once per step, so this costs almost nothing)
  2  in addition the stages inside the propagation and the energy (biorthogonalisation, force bias, auxiliary field,
     Taylor expansion, ...), each synchronised; exact stage times, but with JAX on a GPU the step becomes a few percent
     slower because the device cannot run ahead

Times are wall-clock seconds of every rank; the report gives the maximum, mean and minimum over the ranks (the slowest
rank sets the pace of the run). Operation counts are estimates from the matrix sizes: multiply-adds of the dominant
matrix products, 8 real floating-point operations per complex multiply-add.
"""
import os
import platform
import resource
import socket
import subprocess
import sys
import time

import numpy as np


class _NullSection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def done(self, x):
        return x


_NULL = _NullSection()


class _Section:
    def __init__(self, timers, name):
        self.t, self.name = timers, name

    def __enter__(self):
        t = self.t
        t._stack.append(self.name)
        self.path = "/".join(t._stack)
        t.data.setdefault(self.path, [0, 0.0])                    # registered when entered: run order of the parts
        self.start = time.perf_counter()
        return self

    def done(self, x):
        """Wait until x (array or tuple of arrays) is computed, so that its time is counted in this section (JAX)."""
        if self.t.sync is not None:
            self.t.sync(x)
        return x

    def __exit__(self, *exc):
        t = self.t
        d = t.data.setdefault(self.path, [0, 0.0])
        d[0] += 1
        d[1] += time.perf_counter() - self.start
        t._stack.pop()
        return False


class Timers:
    """Nested wall-clock timers.  with timers.part("energy") as s: ... ; s.done(result)
    part(): level >= 1 (parts of the run); stage(): level >= 2 (stages inside a part)."""

    def __init__(self, level=1, sync=None):
        self.level = int(level)
        self.sync = sync
        self.data = {}
        self._stack = []
        self.t_start = time.perf_counter()

    def part(self, name):
        return _Section(self, name) if self.level >= 1 else _NULL

    def stage(self, name):
        return _Section(self, name) if self.level >= 2 else _NULL


NULL_TIMERS = Timers(level=0)


def jax_sync(x):
    import jax
    jax.block_until_ready(x)
    return x


# ---------------------------------------------------------------------------------------------------------------------
#  operation counts (complex multiply-adds per call, for all walkers of one rank; compact H2 layout)
# ---------------------------------------------------------------------------------------------------------------------
def operation_counts(nk, nb, ne, ng, nmax, nw, order, propagator="S2", nspin=1):
    """nspin = 2 (spin-unrestricted): every count is done once per spin channel."""
    nbk, nek = nk * nb, nk * ne
    biorth = nw * (2 * nbk * nek ** 2 + nek ** 3)                 # trial^T Psi, inverse, Psi (trial^T Psi)^-1
    pq = nw * 2 * nk * nk * nmax * nb * ne                         # P and Q: batched products over q
    aux = nw * 2 * nk * nk * nb * nb * nmax                        # sum_g L u and sum_g conj(L) v
    taylor = nw * order * nbk * nbk * nek                          # field (nbk x nbk) times walker, per order
    h1 = nw * nbk * nbk * nek                                     # one exp(-dtau H1/2) (S2: two calls per step)
    proj = nw * 2 * (nek * nek * nbk + nek ** 3 / 3.0)             # two overlaps: product and determinant
    exch = nw * nk ** 3 * (nb * ne) ** 2                           # Gram form of the exchange
    one = nw * nbk * nek
    chol = nw * (2 * nbk * nek ** 2 + 4.0 / 3.0 * nek ** 3)
    qr = nw * (4 * nbk * nek ** 2 - 4.0 / 3.0 * nek ** 3)
    c = {
        "propagate/biorthogonalize": biorth,
        "propagate/aux_field/force_bias": pq,
        "propagate/aux_field/aux_matrix": aux,
        "propagate/exp_h1": h1,
        "propagate/taylor": taylor,
        "propagate/overlap_weights": proj,
        "energy/biorthogonalize": biorth,
        "energy/one_body": one,
        "energy/hartree": pq,
        "energy/exchange": exch,
        "reortho:Cholesky": chol,
        "reortho:QR": qr,
    }
    c["propagate/aux_field"] = pq + aux
    c["propagate"] = biorth + pq + aux + (2 if propagator == "S2" else 1) * h1 + taylor + proj
    c["energy"] = biorth + one + pq + exch
    return {k: 8.0 * nspin * v for k, v in c.items()}


# ---------------------------------------------------------------------------------------------------------------------
#  environment
# ---------------------------------------------------------------------------------------------------------------------
ENV_KEYS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "XLA_FLAGS",
            "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK"]


def _git_version(path):
    try:
        out = subprocess.run(["git", "-C", path, "describe", "--always", "--dirty"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or "unknown (no git repository)"
    except Exception:
        return "unknown (no git repository)"


def _blas_info():
    lines = []
    try:
        from threadpoolctl import threadpool_info
        for p in threadpool_info():
            lines.append(f"{p.get('internal_api', '?')} {p.get('version', '')}, {p.get('num_threads', '?')} threads"
                         f" ({p.get('user_api', '?')}: {os.path.basename(p.get('filepath', ''))})")
    except Exception:
        pass
    if not lines:
        try:
            cfg = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
            lines.append(f"{cfg.get('name', '?')} {cfg.get('version', '')} (numpy build; threads not available)")
        except Exception:
            lines.append("unknown")
    return lines


def _affinity():
    try:
        return sorted(os.sched_getaffinity(0))
    except Exception:
        return []


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # kB on Linux


def _device_info(backend_name):
    if backend_name.lower() != "jax":
        return ["cpu (no device library)"], None
    try:
        import jax
        devs = jax.devices()
        info = [f"jax {jax.__version__}, default backend {jax.default_backend()}, devices: "
                + ", ".join(f"{d.platform}:{getattr(d, 'device_kind', '')}" for d in devs)]
        peak = None
        try:
            st = devs[0].memory_stats()
            if st:
                peak = st.get("peak_bytes_in_use", 0) / 1024 ** 2
                info.append(f"device memory: peak in use {peak:.0f} MB, limit {st.get('bytes_limit', 0) / 1024 ** 2:.0f} MB")
        except Exception:
            pass
        return info, peak
    except Exception as exc:
        return [f"jax not available ({exc})"], None


# ---------------------------------------------------------------------------------------------------------------------
#  report
# ---------------------------------------------------------------------------------------------------------------------
def _fmt_t(s):
    return f"{s:12.3f}"


def write_report(timers, comm, path, settings, counts=None, code_dir=None, n_steps=None):
    """Gather the timers of all ranks and write the report on rank 0. settings: list of (name, value) of the run.
    counts: operation counts per call of a timer path (operation_counts). Returns summary lines for outcar.txt."""
    rank, size = comm.Get_rank(), comm.Get_size()
    total = time.perf_counter() - timers.t_start
    order = comm.allgather(list(timers.data))                    # run order: first appearance on rank 0, then others
    paths = []
    for lst in order:
        paths += [p for p in lst if p not in paths]
    paths = _tree_order(paths)
    top_self = sum(v[1] for p, v in timers.data.items() if "/" not in p)
    sec = np.array([timers.data.get(p, [0, 0.0])[1] for p in paths] + [total, total - top_self])
    ncalls = np.array([timers.data.get(p, [0, 0.0])[0] for p in paths])
    all_sec = comm.gather(sec, root=0)
    host = socket.gethostname()
    hosts = comm.gather(host, root=0)
    aff = comm.gather(_affinity(), root=0)
    rss = comm.gather(_peak_rss_mb(), root=0)
    backend_name = settings_dict(settings).get("backend", "")
    dev_lines, dev_peak = _device_info(backend_name)
    blas = _blas_info()
    if rank != 0:
        return []

    A = np.array(all_sec)                                          # ranks x (paths + total + not in a part)
    tmax, tmean, tmin = A.max(0), A.mean(0), A.min(0)
    T = tmax[-2]
    from mpi4py import MPI
    mpi_lib = MPI.Get_library_version().strip().splitlines()[0]
    nodes = {}
    for h in hosts:
        nodes[h] = nodes.get(h, 0) + 1
    cores = [len(a) for a in aff]
    shared = len({tuple(a) for a in aff}) < len(aff) if aff and all(aff) else False

    L = []
    w = L.append
    w("=" * 100)
    w("  vafpy timing report")
    w("=" * 100)
    w("")
    w("Code and run")
    w(f"  code                     {code_dir or '?'}")
    w(f"  version (git)            {_git_version(code_dir) if code_dir else 'unknown'}")
    w(f"  python                   {platform.python_version()} ({sys.executable})")
    w(f"  numpy                    {np.__version__}")
    try:
        import scipy
        w(f"  scipy                    {scipy.__version__}")
    except Exception:
        pass
    for k, v in settings:
        w(f"  {k:<24} {v}")
    w("")
    w("Parallel set-up")
    w(f"  MPI library              {mpi_lib}")
    w(f"  MPI ranks                {size} on {len(nodes)} node(s): "
      + ", ".join(f"{h} ({n})" for h, n in sorted(nodes.items())))
    if cores:
        w(f"  CPUs per rank (affinity) min {min(cores)}, max {max(cores)} (hardware threads)"
          + ("; some ranks share the same CPUs" if shared else ""))
        w(f"  CPUs of rank 0           {_ranges(aff[0])}")
    for k in ENV_KEYS:
        if k in os.environ:
            w(f"  {k:<24} {os.environ[k]}")
    for i, b in enumerate(blas):
        w(f"  {'BLAS / threads' if i == 0 else '':<24} {b}")
    for i, d in enumerate(dev_lines):
        w(f"  {'device' if i == 0 else '':<24} {d}")
    w(f"  timing level             {timers.level}" + ("  (stages synchronised)" if timers.level >= 2 else ""))
    w("")
    w("Memory")
    w(f"  peak resident memory per rank: max {max(rss):.0f} MB, mean {np.mean(rss):.0f} MB, min {min(rss):.0f} MB;"
      f" sum over ranks {sum(rss) / 1024:.1f} GB")
    if dev_peak is not None:
        w(f"  device memory of rank 0: peak in use {dev_peak:.0f} MB")
    w("")
    w("Time per part (wall clock, seconds; max / mean / min over the ranks; % of the total run time, max)")
    w(f"  {'part':<40}{'calls':>9}{'max':>12}{'mean':>12}{'min':>12}{'%':>8}{'ms/call':>12}")
    w("  " + "-" * 103)
    for i, p in enumerate(paths):
        depth = p.count("/")
        name = "  " * depth + p.split("/")[-1]
        nc = max(int(ncalls[i]), 1)
        w(f"  {name:<40}{int(ncalls[i]):>9}{_fmt_t(tmax[i])}{_fmt_t(tmean[i])}{_fmt_t(tmin[i])}"
          f"{100 * tmax[i] / T:>8.1f}{1e3 * tmax[i] / nc:>12.3f}")
    w("  " + "-" * 103)
    w(f"  {'not in a part':<40}{'':>9}{_fmt_t(tmax[-1])}{_fmt_t(tmean[-1])}{_fmt_t(tmin[-1])}{100 * tmax[-1] / T:>8.1f}")
    w(f"  {'total run time':<40}{'':>9}{_fmt_t(tmax[-2])}{_fmt_t(tmean[-2])}{_fmt_t(tmin[-2])}")
    top = [i for i, p in enumerate(paths) if "/" not in p]
    rank0_only = [paths[i] for i in top if size > 1 and tmin[i] == 0 and tmax[i] > 0]
    if rank0_only:
        w(f"  done by rank 0 only: {', '.join(rank0_only)}")
    if n_steps:
        loop = [i for i, p in enumerate(paths) if p in ("propagate", "energy", "reortho", "rebalance", "output")]
        w(f"  time per step (parts of the loop, max over ranks): {1e3 * sum(tmax[i] for i in loop) / n_steps:.2f} ms")
    imb = [((tmax[i] - tmean[i]) / tmax[i], paths[i]) for i in top if tmax[i] > 0.01 * T and paths[i] not in rank0_only]
    if imb:
        v, p = max(imb)
        w(f"  load imbalance of the parts above 1 % (all ranks): (max - mean) / max up to {100 * v:.1f} % ({p})")
    flop_lines = []
    if counts:
        w("")
        w("Floating-point operations (estimates from the matrix sizes, 8 flops per complex multiply-add)")
        w(f"  {'part':<40}{'calls':>9}{'GFLOP/call':>12}{'GFLOP/s rank':>14}{'GFLOP/s run':>14}")
        w("  " + "-" * 89)
        tot_f, tot_t = 0.0, 0.0
        for i, p in enumerate(paths):
            key = p if p in counts else None
            if p == "reortho":
                key = next((k for k in counts if k.startswith("reortho:") and k.split(":")[1] in settings_dict(settings).get("reortho", "")), None)
            if key is None or tmax[i] <= 0:
                continue
            fl = counts[key] * ncalls[i]
            name = "  " * p.count("/") + p.split("/")[-1]
            w(f"  {name:<40}{int(ncalls[i]):>9}{counts[key] / 1e9:>12.4f}{fl / tmax[i] / 1e9:>14.2f}"
              f"{size * fl / tmax[i] / 1e9:>14.1f}")
            if "/" not in p:
                tot_f += fl
                tot_t += tmax[i]
        if tot_t > 0:
            w("  " + "-" * 89)
            w(f"  {'counted parts':<40}{'':>9}{'':>12}{tot_f / tot_t / 1e9:>14.2f}{size * tot_f / tot_t / 1e9:>14.1f}")
            flop_lines.append(f"Floating-point rate of the counted parts = {tot_f / tot_t / 1e9:.2f} GFLOP/s per rank,"
                             f" {size * tot_f / tot_t / 1e9:.1f} GFLOP/s in total")
    w("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    lines_out = [f"Timing report = {path} (level {timers.level})",
                 f"Total run time (max over ranks) = {T:.2f} s"] + \
                [f"Time in {paths[i]} = {tmax[i]:.2f} s ({100 * tmax[i] / T:.1f} %)" for i in top] + flop_lines + \
                [f"Peak resident memory per rank = {max(rss):.0f} MB (max), {np.mean(rss):.0f} MB (mean)"]
    return lines_out


def _tree_order(paths):
    """Children directly after their parent, in order of first appearance."""
    out = []

    def add(prefix):
        for p in paths:
            parent = p.rsplit("/", 1)[0] if "/" in p else ""
            if p not in out and parent == prefix:
                out.append(p)
                add(p)
    add("")
    return out + [p for p in paths if p not in out]


def settings_dict(settings):
    return {k: str(v) for k, v in settings}


def _ranges(cores):
    if not cores:
        return "?"
    out, start, prev = [], cores[0], cores[0]
    for c in cores[1:] + [None]:
        if c is None or c != prev + 1:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            if c is not None:
                start = c
        if c is not None:
            prev = c
    return ",".join(out)
