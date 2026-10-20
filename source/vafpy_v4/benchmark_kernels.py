#!/usr/bin/env python3
"""Per-kernel timing and memory of the H2-dependent AFQMC operations.

One process per configuration (layout x backend), so that the peak host memory is meaningful:

    python3 benchmark_kernels.py --data DIR --backend numpy --precision Double --walkers 128 --out bench.json
    python3 benchmark_kernels.py --data DIR --backend jax   --precision Single --walkers 4096 --out bench.json

DIR holds H1_svd.npy, H2_zip.npy, Q_list.npy (and Q_sizes.npy for the compact layout); the layout is detected
from the shape of H2_zip.npy. Kernels are timed on walkers that have been propagated for a few steps, so the
Green's functions are not the trivial HF ones.
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions as new  # noqa: E402
from mpi4py import MPI  # noqa: E402


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def gpu_peak_mb():
    try:
        import jax
        st = jax.devices()[0].memory_stats()
        return None if st is None else st.get("peak_bytes_in_use", 0) / 2**20
    except Exception:
        return None


def nbytes_of(obj):
    """Bytes of all array attributes of a Hamiltonian object (host and device arrays)."""
    out = {}
    for k, v in vars(obj).items():
        if hasattr(v, "nbytes") and getattr(v, "ndim", 0) > 0:
            out[k] = int(v.nbytes)
    return out


def timeit(fn, sync, min_time=0.4, max_rep=200):
    sync(fn())                                   # warm up (JIT, caches)
    t0 = time.perf_counter()
    n = 0
    while True:
        sync(fn())
        n += 1
        dt = time.perf_counter() - t0
        if dt > min_time or n >= max_rep:
            return dt / n * 1000.0               # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "jax"])
    ap.add_argument("--precision", default="Double", choices=["Single", "Double"])
    ap.add_argument("--walkers", type=int, default=128)
    ap.add_argument("--warm-steps", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {"data": a.data, "backend": a.backend, "precision": a.precision, "walkers": a.walkers}
    h1_file = np.load(os.path.join(a.data, "H1_svd.npy"))
    nb, nk = h1_file.shape[0], h1_file.shape[2]
    ql = np.load(os.path.join(a.data, "Q_list.npy"))
    h2_shape = np.load(os.path.join(a.data, "H2_zip.npy"), mmap_mode="r").shape
    ng = h2_shape[2]
    ne = nb // 2                                 # half of the orbitals per k-point are occupied (closed-shell diamond)
    res.update(nb=nb, nk=nk, ne=ne, ng=ng, h2_shape=list(h2_shape))

    backend = new.NumpyBackend(11) if a.backend == "numpy" else new.JaxBackend(11)
    config = new.Configuration(
        num_walkers=a.walkers, num_kpoint=nk, num_orbital=nb, num_electron=ne, num_g=ng, singularity=0.0,
        propagator="S2", order_propagation=6, timestep=2.5e-4, comm=MPI.COMM_WORLD,
        precision=a.precision, backend=backend)
    if a.backend == "jax":
        import jax
        res["device"] = str(jax.devices()[0])
        sync = lambda x: jax.block_until_ready(x)  # noqa: E731
    else:
        sync = lambda x: x                         # noqa: E731

    q_list = new.obtain_Q_list(config, os.path.join(a.data, "Q_list.npy"))
    h2_host, layout = new.obtain_H2_host(config, os.path.join(a.data, "H2_zip.npy"))
    sizes = new.obtain_Q_sizes(config, os.path.join(a.data, "Q_sizes.npy")) if layout == "compact" else None
    res["layout"] = layout
    res["raw_h2_file_MB"] = os.path.getsize(os.path.join(a.data, "H2_zip.npy")) / 2**20
    res["rss_after_load_MB"] = rss_mb()
    h1 = new.obtain_H1(config, os.path.join(a.data, "H1_svd.npy"))
    H = new.build_hamiltonian(config, h1, h2_host, layout, q_list, sizes)
    del h2_host
    trial, walkers = new.initialize_determinant(config)
    t0 = time.perf_counter()
    H.setup_energy_expressions(config, trial)
    res["setup_ms"] = (time.perf_counter() - t0) * 1000
    res["rss_after_setup_MB"] = rss_mb()
    res["hamiltonian_array_bytes"] = nbytes_of(H)
    res["hamiltonian_array_total_MB"] = sum(nbytes_of(H).values()) / 2**20
    res["gpu_peak_after_setup_MB"] = gpu_peak_mb()

    # evolve the walkers a little so that the Green's functions are generic
    e0 = new.measure_energy(config, trial, walkers, H)[0]
    h_0 = np.exp(config.timestep * H.H_zero)
    for _ in range(a.warm_steps):
        walkers, _ = new.propagate_walkers(config, trial, walkers, H, h_0, e0)
        walkers.slater_det = new.reortho_qr(config, walkers.slater_det)
        e0 = new.measure_energy(config, trial, walkers, H)[0]
    theta = new.biorthogonalize(config.backend, trial, walkers.slater_det)
    fb_fun = getattr(H, "_force_bias", None) or H._force_bias_expression
    x = H.create_random_field(config)
    h2_field = H.create_auxiliary_field(config, theta)[0]

    t = {}
    t["biorthogonalize"] = timeit(lambda: new.biorthogonalize(config.backend, trial, walkers.slater_det), sync)
    t["force_bias"] = timeit(lambda: fb_fun(theta), sync)
    t["auxiliary_field"] = timeit(lambda: H._auxiliary_field(x), sync)
    t["create_auxiliary_field(fb+field)"] = timeit(lambda: H.create_auxiliary_field(config, theta)[0], sync)
    t["E1"] = timeit(lambda: H.compute_one_body(theta), sync)
    t["Hartree"] = timeit(lambda: H.compute_hartree(theta), sync)
    t["exchange"] = timeit(lambda: H.compute_exchange(theta), sync)
    t["local_energy(measure_energy)"] = timeit(lambda: new.measure_energy(config, trial, walkers, H)[1], sync)
    half = lambda: H.exp_h1_half @ walkers.slater_det  # noqa: E731
    t["h1_half_application"] = timeit(half, sync)
    t["taylor(exp h2)"] = timeit(lambda: new.apply_taylor(config, h2_field, walkers.slater_det), sync)
    t["propagate_walkers(total)"] = timeit(
        lambda: new.propagate_walkers(config, trial, walkers, H, h_0, e0)[0].slater_det, sync)
    t["reorthogonalise"] = timeit(lambda: new.reortho_qr(config, walkers.slater_det), sync)

    def step():
        w, _ = new.propagate_walkers(config, trial, walkers, H, h_0, e0)
        w.slater_det = new.reortho_qr(config, w.slater_det)
        return new.measure_energy(config, trial, w, H)[1]
    t["total_step(prop+reortho+energy)"] = timeit(step, sync, min_time=1.0, max_rep=50)
    res["kernels_ms"] = t
    res["rss_peak_MB"] = rss_mb()
    res["gpu_peak_MB"] = gpu_peak_mb()
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
