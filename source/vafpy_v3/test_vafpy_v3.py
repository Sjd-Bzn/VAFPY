"""End-to-end verification of vafpy_v2 against the reference opt code.

The test is deterministic: we feed the propagator a *fixed* random
field via Hamiltonian.test_random_field so the new propagator output
can be compared against an independent re-implementation of the opt
reference equations.
"""
import os
import sys
import numpy as np
from cmath import phase
from scipy.linalg import expm, block_diag

sys.path.insert(0, os.path.dirname(__file__))
import functions as new


def build_config(num_k, num_orb, num_e, num_g, num_walkers, dtau, propagator="S2"):
    backend = new.NumpyBackend(seed=12345)
    from mpi4py import MPI
    return new.Configuration(
        num_walkers=num_walkers,
        num_kpoint=num_k,
        num_orbital=num_orb,
        num_electron=num_e,
        num_g=num_g,
        singularity=0.0,
        propagator=propagator,
        order_propagation=6,
        timestep=dtau,
        comm=MPI.COMM_WORLD,
        precision="Double",
        backend=backend,
    )


# --------------------------------------------------------------------------
# Reference re-implementation (mirrors opt update_hyb / measure_E_gs).
# --------------------------------------------------------------------------
def reference_propagate(trial, walker, weight, h2_e, h2_o, h1_total,
                        H_zero, dtau, e0, x_e, x_o, order_trunc):
    """Single-walker S2 propagator following Code/opt/afqmc_func_ref.py."""
    sqrt_dtau = np.sqrt(dtau)
    th = walker @ np.linalg.inv(trial.T @ walker)
    alpha_e = np.einsum("pi, prg -> irg", trial, h2_e)
    alpha_o = np.einsum("pi, prg -> irg", trial, h2_o)
    fb_e = -2j * sqrt_dtau * np.einsum("ri, irG -> G", th, alpha_e)
    fb_o = -2j * sqrt_dtau * np.einsum("ri, irG -> G", th, alpha_o)
    fb_e[abs(fb_e) >= 1] = 0
    fb_o[abs(fb_o) >= 1] = 0

    h2 = h2_e @ (x_e - fb_e) + h2_o @ (x_o - fb_o)

    h1_half = expm(-dtau * h1_total / 2)

    # Taylor exp(1j * sqrt_dtau * h2)
    mat = 1j * sqrt_dtau * h2
    eye = np.eye(mat.shape[0], dtype=np.complex128)
    out = eye.copy()
    c = eye.copy()
    for i in range(order_trunc):
        c = mat @ c / (i + 1)
        out += c
    h2_exp = out

    prop = np.exp(dtau * H_zero) * h1_half @ h2_exp @ h1_half
    new_walker = prop @ walker

    ovr_old = np.linalg.det(trial.T @ walker) ** 2
    ovr_new = np.linalg.det(trial.T @ new_walker) ** 2
    overlap_ratio = ovr_new / ovr_old
    alpha = phase(overlap_ratio)

    importance = np.exp(np.dot(x_e, fb_e) - np.dot(fb_e, fb_e) / 2) \
        * np.exp(np.dot(x_o, fb_o) - np.dot(fb_o, fb_o) / 2)
    a = overlap_ratio * importance * np.exp(dtau * e0)
    factor = abs(a) if abs(a) < 10 else 0
    new_weight = factor * max(0, np.cos(alpha)) * weight
    return new_walker, new_weight


# --------------------------------------------------------------------------
def test_hf_energy_matches_opt():
    """HF energy components should match Code/opt to 1e-10."""
    config = build_config(num_k=1, num_orb=8, num_e=4, num_g=36,
                          num_walkers=10, dtau=0.005)
    ql = new.obtain_Q_list(config, "Q_list.npy")
    h1 = new.obtain_H1(config, "H1_svd.npy")
    h2 = new.obtain_H2(config, "H2_zip.npy")
    H = new.Hamiltonian(one_body=h1, two_body=h2, q_list=ql)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)

    e1, eh, ex = new.measure_components(config, trial, walkers, H)
    e_tot, _, _ = new.measure_energy(config, trial, walkers, H)

    ref = {
        "E_one": 101.97815646937238,
        "Hartree": 38.0138379353566,
        "Exchange": -25.91436338445311,
        "Total": 114.0776310202759,
        "H_zero": 4.751729741919572,
    }
    assert abs(e1.real - ref["E_one"]) < 1e-9, (e1, ref["E_one"])
    assert abs(eh.real - ref["Hartree"]) < 1e-9, (eh, ref["Hartree"])
    assert abs(ex.real - ref["Exchange"]) < 1e-9, (ex, ref["Exchange"])
    assert abs(e_tot.real - ref["Total"]) < 1e-9, (e_tot, ref["Total"])
    assert abs(H.H_zero.real - ref["H_zero"]) < 1e-9
    print("[PASS] HF energy components match opt reference exactly.")


def test_propagator_matches_opt_with_fixed_field():
    """Per-walker S2 propagation matches the opt reference for fixed x_e, x_o."""
    num_walkers = 5
    num_g = 36
    config = build_config(num_k=1, num_orb=8, num_e=4, num_g=num_g,
                          num_walkers=num_walkers, dtau=0.005)
    ql = new.obtain_Q_list(config, "Q_list.npy")
    h1 = new.obtain_H1(config, "H1_svd.npy")
    h2 = new.obtain_H2(config, "H2_zip.npy")
    H = new.Hamiltonian(one_body=h1, two_body=h2, q_list=ql)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)

    rng = np.random.default_rng(202)
    walkers.slater_det = walkers.slater_det + 0.01 * rng.standard_normal(
        walkers.slater_det.shape
    ).astype(np.complex128)
    walkers.weights = walkers.weights * (1 + 0.01 * rng.standard_normal(num_walkers))

    # Fixed random field — same layout the new code expects.
    rng = np.random.default_rng(31415)
    x_e = rng.standard_normal((num_g, num_walkers))
    x_o = rng.standard_normal((num_g, num_walkers))
    H.test_random_field = np.concatenate(
        (x_e.astype(np.complex128), x_o.astype(np.complex128)), axis=0
    )

    e_hf, _, _ = new.measure_energy(config, trial, walkers, H)
    h_0 = np.exp(config.timestep * H.H_zero)
    new_walkers, num_rare = new.propagate_walkers(
        config, trial, walkers, H, h_0, e_hf
    )

    # Recompute h2_e / h2_o on host for the reference path.
    h1_host = config.backend.to_numpy(h1)
    h2_host = config.backend.to_numpy(h2)
    h1_total = (config.backend.to_numpy(H.exp_h1)).copy()  # not used directly
    # h1_total (Hermitian generator with mean-field subtraction) is hidden in
    # exp_h1; recover it via -log(exp_h1)/timestep — too lossy. Instead rebuild
    # using the same helpers as functions.py.
    trial_host = config.backend.to_numpy(trial)
    h2_dag = np.einsum("prG->rpG", h2_host.conj())
    h_mf = new.H_1_mf(trial_host, trial_host, h2_host, h2_dag, ql,
                      h1_host, config.num_kpoint,
                      config.num_orbital, config.num_electron)
    h_sic = -np.einsum("ijG, kjG -> ik", h2_host, h2_dag) / 2
    h1_full = h_mf + h_sic
    h2_mf = new.A_af_MF_sub(trial_host, trial_host, h2_host, ql,
                            config.num_kpoint, config.num_orbital,
                            config.num_electron)
    h2_e = new.gen_A_e(h2_mf)
    h2_o = new.gen_A_o(h2_mf)

    # Mirror the new propagator behaviour for each walker.
    max_det_err = 0.0
    max_w_err = 0.0
    for i in range(num_walkers):
        w_old = config.backend.to_numpy(walkers.slater_det)[i]
        weight_old = complex(config.backend.to_numpy(walkers.weights)[i])
        ref_walker, ref_weight = reference_propagate(
            trial_host, w_old, weight_old,
            h2_e, h2_o, h1_full,
            H.H_zero, config.timestep, e_hf,
            x_e[:, i], x_o[:, i], config.order_propagation,
        )
        got_walker = config.backend.to_numpy(new_walkers.slater_det)[i]
        got_weight = complex(config.backend.to_numpy(new_walkers.weights)[i])
        max_det_err = max(max_det_err,
                          np.max(np.abs(got_walker - ref_walker)))
        max_w_err = max(max_w_err, abs(got_weight - ref_weight))

    assert max_det_err < 1e-8, max_det_err
    assert max_w_err < 1e-8, max_w_err
    print(f"[PASS] S2 propagator matches opt reference "
          f"(det err {max_det_err:.2e}, weight err {max_w_err:.2e}).")


def test_initialize_multi_k_trial():
    """For num_k > 1 the trial should be block diagonal of the single-k eye."""
    config = build_config(num_k=4, num_orb=8, num_e=4, num_g=36,
                          num_walkers=2, dtau=0.005)
    trial, walkers = new.initialize_determinant(config)
    trial_host = config.backend.to_numpy(trial)
    assert trial_host.shape == (32, 16), trial_host.shape
    # off-block elements must be zero
    for k1 in range(4):
        for k2 in range(4):
            block = trial_host[k1 * 8:(k1 + 1) * 8, k2 * 4:(k2 + 1) * 4]
            if k1 == k2:
                assert np.allclose(block, np.eye(8, 4))
            else:
                assert np.allclose(block, 0)
    # walkers are independent copies
    sd = config.backend.to_numpy(walkers.slater_det)
    assert sd.shape == (2, 32, 16)
    assert np.allclose(sd[0], sd[1])
    print("[PASS] multi-k trial determinant is properly block diagonal.")


def test_qlist_loading_and_default():
    """Q_list.npy is consumed; default heuristic also works."""
    config = build_config(num_k=1, num_orb=8, num_e=4, num_g=36,
                          num_walkers=2, dtau=0.005)
    ql = new.obtain_Q_list(config, "Q_list.npy")
    assert ql.shape[1] == 3, ql.shape
    # heuristic fallback should produce something non-empty
    default_ql = new.build_default_q_list(4)
    assert default_ql.shape[1] == 3
    assert default_ql.shape[0] >= 4
    print("[PASS] Q_list loading + default generation.")


def test_multi_k_consistency():
    """Stack a single-k Hamiltonian into a synthetic num_k=2 setup and check
    that the HF energy doubles (per-k average remains the same).

    For block-diagonal H1 / H2 (no inter-k coupling), running with num_k=2
    must give 2× the single-k energy components before the /num_k normalization,
    and an identical total energy after dividing by num_k.
    """
    cfg1 = build_config(num_k=1, num_orb=8, num_e=4, num_g=36,
                        num_walkers=5, dtau=0.005)
    ql1 = new.obtain_Q_list(cfg1, "Q_list.npy")
    h1_1 = new.obtain_H1(cfg1, "H1_svd.npy")
    h2_1 = new.obtain_H2(cfg1, "H2_zip.npy")
    H1ham = new.Hamiltonian(one_body=h1_1, two_body=h2_1, q_list=ql1)
    trial1, walkers1 = new.initialize_determinant(cfg1)
    H1ham.setup_energy_expressions(cfg1, trial1)
    e1_total, _, _ = new.measure_energy(cfg1, trial1, walkers1, H1ham)

    # Build synthetic two-k inputs (independent copy of the same k-block).
    h1_per_k = np.load("H1_svd.npy")  # (8, 8, 1)
    h1_two = np.zeros((8, 8, 2), dtype=h1_per_k.dtype)
    h1_two[:, :, 0] = h1_per_k[:, :, 0]
    h1_two[:, :, 1] = h1_per_k[:, :, 0]
    np.save("/tmp/H1_two.npy", h1_two)

    h2_one = np.load("H2_zip.npy")  # (8, 8, 36)
    # Build block-diagonal (16, 16, 36*2 = 72) — independent fields per k.
    nb, _, ng = h2_one.shape
    h2_two = np.zeros((nb * 2, nb * 2, ng * 2), dtype=h2_one.dtype)
    h2_two[:nb, :nb, :ng] = h2_one
    h2_two[nb:, nb:, ng:] = h2_one
    np.save("/tmp/H2_two.npy", h2_two)

    ql_two = np.array([[1, 1, 1], [2, 2, 1]], dtype=np.int64)
    np.save("/tmp/Q_two.npy", ql_two.T)

    cfg2 = build_config(num_k=2, num_orb=8, num_e=4, num_g=72,
                        num_walkers=5, dtau=0.005)
    ql2 = new.obtain_Q_list(cfg2, "/tmp/Q_two.npy")
    h1_2 = new.obtain_H1(cfg2, "/tmp/H1_two.npy")
    h2_2 = new.obtain_H2(cfg2, "/tmp/H2_two.npy")
    H2ham = new.Hamiltonian(one_body=h1_2, two_body=h2_2, q_list=ql2)
    trial2, walkers2 = new.initialize_determinant(cfg2)
    H2ham.setup_energy_expressions(cfg2, trial2)
    e2_total, _, _ = new.measure_energy(cfg2, trial2, walkers2, H2ham)

    # Energy per k-point must match the single-k computation.
    err = abs(e2_total.real - e1_total.real)
    assert err < 1e-7, (e2_total, e1_total, err)
    print(f"[PASS] multi-k (num_k=2) reproduces single-k energy after /num_k "
          f"(err {err:.2e}).")


def main():
    print("Running vafpy_v2 verification tests...\n")
    test_hf_energy_matches_opt()
    test_initialize_multi_k_trial()
    test_qlist_loading_and_default()
    test_propagator_matches_opt_with_fixed_field()
    test_multi_k_consistency()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
