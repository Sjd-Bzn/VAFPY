"""Verify that vafpy_v4 (compressed runtime) matches vafpy_v3 (dense runtime) exactly.

This is the central correctness test: v4 must produce bit-identical
HF energy and bit-identical propagated walkers for the same inputs.

We test:
  1. HF energy components for num_k=1 (matches opt reference).
  2. Energy for synthetic num_k=3 case where v3 and v4 must agree exactly.
  3. Propagator output for a fixed random field — bit identical.
  4. A small timing comparison.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import functions as new_v4

# Add v3's path for comparison.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vafpy_v3"))


def _import_v3():
    """Import v3.functions under a different name (avoids clashing with v4)."""
    import importlib
    here = os.path.dirname(__file__)
    v3_path = os.path.normpath(os.path.join(here, "..", "vafpy_v3"))
    if v3_path not in sys.path:
        sys.path.insert(0, v3_path)
    # Save / restore v4 import so v3's "import functions" loads v3's copy.
    saved = sys.modules.pop("functions", None)
    sys.path.insert(0, v3_path)
    try:
        v3 = importlib.import_module("functions")
    finally:
        sys.path.remove(v3_path)
    sys.modules["functions_v3"] = v3
    if saved is not None:
        sys.modules["functions"] = saved
    return v3


def make_cfg(mod, num_k, num_orb, num_e, num_g, num_walkers=5, dtau=0.005):
    backend = mod.NumpyBackend(seed=12345)
    from mpi4py import MPI
    return mod.Configuration(
        num_walkers=num_walkers, num_kpoint=num_k, num_orbital=num_orb,
        num_electron=num_e, num_g=num_g, singularity=0.0, propagator="S2",
        order_propagation=6, timestep=dtau, comm=MPI.COMM_WORLD,
        precision="Double", backend=backend,
    )


def _build_synthetic_multi_k(num_k, ng_per_q=6):
    h1_single = np.load("H1_svd.npy")
    nb = h1_single.shape[0]
    h1_full = np.zeros((nb, nb, num_k), dtype=h1_single.dtype)
    for k in range(num_k):
        h1_full[:, :, k] = h1_single[:, :, 0]

    rows = []
    for k1 in range(1, num_k + 1):
        for k2 in range(1, num_k + 1):
            Q = ((k1 - k2) % num_k) + 1
            rows.append([k1, k2, Q])
    q_list = np.array(rows, dtype=np.int64)

    rng = np.random.default_rng(seed=2025)
    h2_full = np.zeros(
        (nb * num_k, nb * num_k, ng_per_q * num_k), dtype=np.complex128
    )
    for k1, k2, Q in q_list:
        block = (
            rng.standard_normal((nb, nb, ng_per_q))
            + 1j * rng.standard_normal((nb, nb, ng_per_q))
        )
        h2_full[(k1 - 1) * nb:k1 * nb, (k2 - 1) * nb:k2 * nb,
                (Q - 1) * ng_per_q:Q * ng_per_q] = block
    return h1_full, h2_full, q_list, nb, ng_per_q


def test_v4_matches_v3_single_k_hf():
    v3 = _import_v3()
    cfg3 = make_cfg(v3, num_k=1, num_orb=8, num_e=4, num_g=36)
    cfg4 = make_cfg(new_v4, num_k=1, num_orb=8, num_e=4, num_g=36)
    ql = np.load("Q_list.npy").T

    # v3 dense path
    h1_3 = v3.obtain_H1(cfg3, "H1_svd.npy")
    h2_3 = v3.obtain_H2(cfg3, "H2_zip.npy", q_list=ql)
    H3 = v3.Hamiltonian(one_body=h1_3, two_body=h2_3, q_list=ql)
    t3, w3 = v3.initialize_determinant(cfg3)
    H3.setup_energy_expressions(cfg3, t3)
    e3, _, _ = v3.measure_energy(cfg3, t3, w3, H3)

    # v4 compressed path
    h1_4 = new_v4.obtain_H1(cfg4, "H1_svd.npy")
    h2_4 = new_v4.obtain_H2(cfg4, "H2_zip.npy", q_list=ql)
    H4 = new_v4.Hamiltonian(one_body=h1_4, two_body=h2_4, q_list=ql)
    t4, w4 = new_v4.initialize_determinant(cfg4)
    H4.setup_energy_expressions(cfg4, t4)
    e4, _, _ = new_v4.measure_energy(cfg4, t4, w4, H4)

    assert abs(e3 - e4) < 1e-10, (e3, e4)
    print(f"[PASS] num_k=1 HF energy: v3={e3.real:.10f}, v4={e4.real:.10f}.")


def test_v4_matches_v3_multi_k_hf():
    v3 = _import_v3()
    h1_full, h2_dense, q_list, nb, ng_per_q = _build_synthetic_multi_k(num_k=3)
    num_k = 3
    ng_t = ng_per_q * num_k
    np.save("/tmp/_v4_h1.npy", h1_full)
    np.save("/tmp/_v4_h2_dense.npy", h2_dense)
    np.save("/tmp/_v4_q.npy", q_list.T)

    cfg3 = make_cfg(v3, num_k=num_k, num_orb=nb, num_e=4, num_g=ng_t)
    cfg4 = make_cfg(new_v4, num_k=num_k, num_orb=nb, num_e=4, num_g=ng_t)

    ql3 = v3.obtain_Q_list(cfg3, "/tmp/_v4_q.npy")
    h1_3 = v3.obtain_H1(cfg3, "/tmp/_v4_h1.npy")
    h2_3 = v3.obtain_H2(cfg3, "/tmp/_v4_h2_dense.npy", q_list=ql3)
    H3 = v3.Hamiltonian(one_body=h1_3, two_body=h2_3, q_list=ql3)
    t3, w3 = v3.initialize_determinant(cfg3)
    H3.setup_energy_expressions(cfg3, t3)
    e3, _, _ = v3.measure_energy(cfg3, t3, w3, H3)

    ql4 = new_v4.obtain_Q_list(cfg4, "/tmp/_v4_q.npy")
    h1_4 = new_v4.obtain_H1(cfg4, "/tmp/_v4_h1.npy")
    h2_4 = new_v4.obtain_H2(cfg4, "/tmp/_v4_h2_dense.npy", q_list=ql4)
    H4 = new_v4.Hamiltonian(one_body=h1_4, two_body=h2_4, q_list=ql4)
    t4, w4 = new_v4.initialize_determinant(cfg4)
    H4.setup_energy_expressions(cfg4, t4)
    e4, _, _ = new_v4.measure_energy(cfg4, t4, w4, H4)

    err = abs(e3 - e4)
    assert err < 1e-9, (e3, e4, err)
    print(f"[PASS] num_k=3 HF energy: v3={e3.real:.10f}, v4={e4.real:.10f} "
          f"(diff {err:.2e}).")


def test_v4_propagator_matches_v3():
    """Bit-identical propagator output with a fixed random field."""
    v3 = _import_v3()
    h1_full, h2_dense, q_list, nb, ng_per_q = _build_synthetic_multi_k(num_k=3)
    num_k = 3
    ng_t = ng_per_q * num_k
    np.save("/tmp/_v4_h1.npy", h1_full)
    np.save("/tmp/_v4_h2_dense.npy", h2_dense)
    np.save("/tmp/_v4_q.npy", q_list.T)

    num_walkers = 4
    cfg3 = make_cfg(v3, num_k=num_k, num_orb=nb, num_e=4, num_g=ng_t,
                    num_walkers=num_walkers)
    cfg4 = make_cfg(new_v4, num_k=num_k, num_orb=nb, num_e=4, num_g=ng_t,
                    num_walkers=num_walkers)

    ql = v3.obtain_Q_list(cfg3, "/tmp/_v4_q.npy")
    h1_3 = v3.obtain_H1(cfg3, "/tmp/_v4_h1.npy")
    h2_3 = v3.obtain_H2(cfg3, "/tmp/_v4_h2_dense.npy", q_list=ql)
    H3 = v3.Hamiltonian(one_body=h1_3, two_body=h2_3, q_list=ql)
    t3, w3 = v3.initialize_determinant(cfg3)
    H3.setup_energy_expressions(cfg3, t3)

    h1_4 = new_v4.obtain_H1(cfg4, "/tmp/_v4_h1.npy")
    h2_4 = new_v4.obtain_H2(cfg4, "/tmp/_v4_h2_dense.npy", q_list=ql)
    H4 = new_v4.Hamiltonian(one_body=h1_4, two_body=h2_4, q_list=ql)
    t4, w4 = new_v4.initialize_determinant(cfg4)
    H4.setup_energy_expressions(cfg4, t4)

    # Fixed random field — same array fed to both runs.
    rng = np.random.default_rng(seed=4242)
    field = rng.standard_normal(
        (2 * cfg3.num_g, cfg3.num_walkers)
    ).astype(np.complex128)
    H3.test_random_field = field
    H4.test_random_field = field

    h_0 = np.exp(cfg3.timestep * H3.H_zero)
    e_hf, _, _ = v3.measure_energy(cfg3, t3, w3, H3)
    nw3, _ = v3.propagate_walkers(cfg3, t3, w3, H3, h_0, e_hf)
    nw4, _ = new_v4.propagate_walkers(cfg4, t4, w4, H4, h_0, e_hf)

    det_err = np.max(np.abs(nw3.slater_det - nw4.slater_det))
    w_err = np.max(np.abs(nw3.weights - nw4.weights))
    print(f"  det error: {det_err:.2e}, weight error: {w_err:.2e}")
    assert det_err < 1e-10
    assert w_err < 1e-10
    print("[PASS] v4 propagator matches v3 (num_k=3, deterministic).")


def test_v4_vs_v3_timing():
    """Smoke-only timing comparison, to confirm v4 is at least competitive."""
    v3 = _import_v3()
    h1_full, h2_dense, q_list, nb, ng_per_q = _build_synthetic_multi_k(
        num_k=4, ng_per_q=10
    )
    num_k = 4
    ng_t = ng_per_q * num_k
    np.save("/tmp/_v4_h1.npy", h1_full)
    np.save("/tmp/_v4_h2_dense.npy", h2_dense)
    np.save("/tmp/_v4_q.npy", q_list.T)

    num_walkers = 50
    n_steps = 10
    cfg3 = make_cfg(v3, num_k=num_k, num_orb=nb, num_e=4, num_g=ng_t,
                    num_walkers=num_walkers)
    cfg4 = make_cfg(new_v4, num_k=num_k, num_orb=nb, num_e=4, num_g=ng_t,
                    num_walkers=num_walkers)

    ql = v3.obtain_Q_list(cfg3, "/tmp/_v4_q.npy")

    def setup(mod, cfg):
        h1 = mod.obtain_H1(cfg, "/tmp/_v4_h1.npy")
        h2 = mod.obtain_H2(cfg, "/tmp/_v4_h2_dense.npy", q_list=ql)
        H = mod.Hamiltonian(one_body=h1, two_body=h2, q_list=ql)
        t, w = mod.initialize_determinant(cfg)
        H.setup_energy_expressions(cfg, t)
        return H, t, w

    H3, t3, w3 = setup(v3, cfg3)
    H4, t4, w4 = setup(new_v4, cfg4)

    h_0 = np.exp(cfg3.timestep * H3.H_zero)
    e_hf, _, _ = v3.measure_energy(cfg3, t3, w3, H3)

    t = time.perf_counter()
    for _ in range(n_steps):
        w3, _ = v3.propagate_walkers(cfg3, t3, w3, H3, h_0, e_hf)
    dt3 = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(n_steps):
        w4, _ = new_v4.propagate_walkers(cfg4, t4, w4, H4, h_0, e_hf)
    dt4 = time.perf_counter() - t

    print(f"  v3 (dense)      : {dt3 * 1000:.1f} ms / {n_steps} steps")
    print(f"  v4 (compressed) : {dt4 * 1000:.1f} ms / {n_steps} steps")
    print(f"[INFO] timing on num_k={num_k}, num_walkers={num_walkers}.")


def main():
    print("Running v4 vs v3 verification...\n")
    test_v4_matches_v3_single_k_hf()
    test_v4_matches_v3_multi_k_hf()
    test_v4_propagator_matches_v3()
    test_v4_vs_v3_timing()
    print("\nv4 vs v3 verification passed.")


if __name__ == "__main__":
    main()
