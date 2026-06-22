"""vafpy_v3 — verify the (nb*nk, nb, ng*nk) compressed-H2 storage path.

Checks:
  * compress/expand round-trip is lossless.
  * obtain_H2 auto-detects the compressed layout.
  * HF energy from a compressed file equals the dense baseline (bit-identical).
  * Propagator output is bit-identical with a fixed random field.
  * build_k2_map rejects ambiguous (non-momentum-conserving) Q-lists.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import functions as new


def make_config(num_k, num_orb, num_e, num_g, num_walkers=5, dtau=0.005):
    backend = new.NumpyBackend(seed=12345)
    from mpi4py import MPI
    return new.Configuration(
        num_walkers=num_walkers, num_kpoint=num_k, num_orbital=num_orb,
        num_electron=num_e, num_g=num_g, singularity=0.0, propagator="S2",
        order_propagation=6, timestep=dtau, comm=MPI.COMM_WORLD,
        precision="Double", backend=backend,
    )


def _build_synthetic_multi_k(num_k, ng_per_q=6):
    """Synthetic multi-k Hamiltonian with cyclic momentum conservation.

    Cyclic Q-list: Q = ((k1 - k2) mod nk) + 1, gives unique (k1, Q) → k2.
    """
    h1_single = np.load("H1_svd.npy")  # (8, 8, 1)
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


def test_k2_map_against_q_list():
    rows = []
    for k1 in range(1, 4):
        for k2 in range(1, 4):
            Q = ((k1 - k2) % 3) + 1
            rows.append([k1, k2, Q])
    q_list = np.array(rows, dtype=np.int64)
    k2_map = new.build_k2_map(q_list, num_k=3)
    for k1 in range(1, 4):
        for Q in range(1, 4):
            expected = ((k1 - 1) - (Q - 1)) % 3
            assert k2_map[k1 - 1, Q - 1] == expected, (k1, Q)
    print("[PASS] build_k2_map (cyclic Q-list).")


def test_k2_map_rejects_ambiguous_qlist():
    bad = np.array([[2, 1, 2], [2, 3, 2]], dtype=np.int64)
    try:
        new.build_k2_map(bad, num_k=3)
    except ValueError as exc:
        assert "multiple k2" in str(exc)
        print("[PASS] build_k2_map rejects ambiguous Q-list.")
        return
    raise AssertionError("expected ValueError")


def test_compress_expand_roundtrip_num_k_1():
    h2 = np.load("H2_zip.npy")
    q_list = np.load("Q_list.npy").T
    nb, _, ng = h2.shape
    h2_c = new.compress_h2(h2, q_list, 1, nb, ng)
    h2_d = new.expand_h2(h2_c, q_list, 1, nb, ng)
    assert np.array_equal(h2, h2_d)
    print("[PASS] compress/expand round-trip (num_k=1).")


def test_compress_expand_roundtrip_multi_k():
    _, h2_dense, q_list, nb, ng_per_q = _build_synthetic_multi_k(num_k=3)
    num_k = 3
    ng_tot = ng_per_q * num_k
    h2_c = new.compress_h2(h2_dense, q_list, num_k, nb, ng_tot)
    assert h2_c.shape == (nb * num_k, nb, ng_tot), h2_c.shape
    h2_back = new.expand_h2(h2_c, q_list, num_k, nb, ng_tot)
    assert np.array_equal(h2_dense, h2_back), \
        f"diff max = {np.max(np.abs(h2_dense - h2_back))}"
    print("[PASS] compress/expand round-trip (num_k=3 synthetic).")


def test_obtain_h2_autodetect():
    h1_full, h2_dense, q_list, nb, ng_per_q = _build_synthetic_multi_k(num_k=2)
    num_k = 2
    ng_tot = ng_per_q * num_k

    np.save("/tmp/_v3_h2_dense.npy", h2_dense)
    np.save(
        "/tmp/_v3_h2_compressed.npy",
        new.compress_h2(h2_dense, q_list, num_k, nb, ng_tot),
    )
    np.save("/tmp/_v3_h1.npy", h1_full)
    np.save("/tmp/_v3_q.npy", q_list.T)

    cfg = make_config(num_k=num_k, num_orb=nb, num_e=4, num_g=ng_tot)
    ql = new.obtain_Q_list(cfg, "/tmp/_v3_q.npy")
    h2d = new.obtain_H2(cfg, "/tmp/_v3_h2_dense.npy", q_list=ql)
    h2c = new.obtain_H2(cfg, "/tmp/_v3_h2_compressed.npy", q_list=ql)
    assert np.array_equal(
        cfg.backend.to_numpy(h2d), cfg.backend.to_numpy(h2c)
    )
    print("[PASS] obtain_H2 auto-detects the compressed format.")


def test_compressed_hf_energy_single_k():
    cfg = make_config(num_k=1, num_orb=8, num_e=4, num_g=36)
    ql = new.obtain_Q_list(cfg, "Q_list.npy")
    h1 = new.obtain_H1(cfg, "H1_svd.npy")

    h2_d = new.obtain_H2(cfg, "H2_zip.npy", q_list=ql)
    H_d = new.Hamiltonian(one_body=h1, two_body=h2_d, q_list=ql)
    trial, walkers = new.initialize_determinant(cfg)
    H_d.setup_energy_expressions(cfg, trial)
    e_d, _, _ = new.measure_energy(cfg, trial, walkers, H_d)

    new.save_h2_compressed(
        np.load("H2_zip.npy"), ql, cfg.num_kpoint, cfg.num_orbital,
        cfg.num_g, "/tmp/_v3_h2_zip_c.npy",
    )
    h2_c = new.obtain_H2(cfg, "/tmp/_v3_h2_zip_c.npy", q_list=ql)
    H_c = new.Hamiltonian(one_body=h1, two_body=h2_c, q_list=ql)
    trial2, walkers2 = new.initialize_determinant(cfg)
    H_c.setup_energy_expressions(cfg, trial2)
    e_c, _, _ = new.measure_energy(cfg, trial2, walkers2, H_c)
    assert e_d == e_c, (e_d, e_c)
    print(f"[PASS] compressed HF energy bit-identical to dense "
          f"(E={e_d.real:.10f}).")


def test_compressed_hf_energy_multi_k():
    h1_full, h2_dense, q_list, nb, ng_per_q = _build_synthetic_multi_k(num_k=2)
    num_k = 2
    ng_tot = ng_per_q * num_k
    np.save("/tmp/_v3_h1_two.npy", h1_full)
    np.save("/tmp/_v3_h2_two_d.npy", h2_dense)
    np.save(
        "/tmp/_v3_h2_two_c.npy",
        new.compress_h2(h2_dense, q_list, num_k, nb, ng_tot),
    )
    np.save("/tmp/_v3_q_two.npy", q_list.T)

    cfg = make_config(num_k=num_k, num_orb=nb, num_e=4, num_g=ng_tot)
    ql = new.obtain_Q_list(cfg, "/tmp/_v3_q_two.npy")
    h1 = new.obtain_H1(cfg, "/tmp/_v3_h1_two.npy")
    h2_d = new.obtain_H2(cfg, "/tmp/_v3_h2_two_d.npy", q_list=ql)
    h2_c = new.obtain_H2(cfg, "/tmp/_v3_h2_two_c.npy", q_list=ql)

    H_d = new.Hamiltonian(one_body=h1, two_body=h2_d, q_list=ql)
    H_c = new.Hamiltonian(one_body=h1, two_body=h2_c, q_list=ql)
    trial, walkers = new.initialize_determinant(cfg)
    trial2, walkers2 = new.initialize_determinant(cfg)
    H_d.setup_energy_expressions(cfg, trial)
    H_c.setup_energy_expressions(cfg, trial2)
    e_d, _, _ = new.measure_energy(cfg, trial, walkers, H_d)
    e_c, _, _ = new.measure_energy(cfg, trial2, walkers2, H_c)
    assert e_d == e_c, (e_d, e_c)
    print(f"[PASS] compressed HF energy (num_k=2) matches dense "
          f"(E={e_d.real:.10f}).")


def test_compressed_propagator_matches_dense():
    cfg = make_config(num_k=1, num_orb=8, num_e=4, num_g=36, num_walkers=4)
    ql = new.obtain_Q_list(cfg, "Q_list.npy")
    h1 = new.obtain_H1(cfg, "H1_svd.npy")

    h2_d = new.obtain_H2(cfg, "H2_zip.npy", q_list=ql)
    H_d = new.Hamiltonian(one_body=h1, two_body=h2_d, q_list=ql)
    trial, walkers = new.initialize_determinant(cfg)
    H_d.setup_energy_expressions(cfg, trial)

    new.save_h2_compressed(
        np.load("H2_zip.npy"), ql, cfg.num_kpoint, cfg.num_orbital,
        cfg.num_g, "/tmp/_v3_h2_zip_c.npy",
    )
    h2_c = new.obtain_H2(cfg, "/tmp/_v3_h2_zip_c.npy", q_list=ql)
    H_c = new.Hamiltonian(one_body=h1, two_body=h2_c, q_list=ql)
    trial2, walkers2 = new.initialize_determinant(cfg)
    H_c.setup_energy_expressions(cfg, trial2)

    rng = np.random.default_rng(2027)
    field = rng.standard_normal(
        (2 * cfg.num_g, cfg.num_walkers)
    ).astype(np.complex128)
    H_d.test_random_field = field
    H_c.test_random_field = field

    h_0 = np.exp(cfg.timestep * H_d.H_zero)
    e_hf, _, _ = new.measure_energy(cfg, trial, walkers, H_d)

    nw_d, _ = new.propagate_walkers(cfg, trial, walkers, H_d, h_0, e_hf)
    nw_c, _ = new.propagate_walkers(cfg, trial2, walkers2, H_c, h_0, e_hf)
    det_err = np.max(np.abs(nw_d.slater_det - nw_c.slater_det))
    w_err = np.max(np.abs(nw_d.weights - nw_c.weights))
    assert det_err == 0 and w_err == 0
    print("[PASS] compressed propagator bit-identical "
          f"(det err {det_err:.2e}, weight err {w_err:.2e}).")


def main():
    print("Running vafpy_v3 compressed-H2 verification...\n")
    test_k2_map_against_q_list()
    test_k2_map_rejects_ambiguous_qlist()
    test_compress_expand_roundtrip_num_k_1()
    test_compress_expand_roundtrip_multi_k()
    test_obtain_h2_autodetect()
    test_compressed_hf_energy_single_k()
    test_compressed_hf_energy_multi_k()
    test_compressed_propagator_matches_dense()
    print("\nAll vafpy_v3 compressed-H2 tests passed.")


if __name__ == "__main__":
    main()
