"""theta, overlap and P, Q kept with the walkers (vafpy_v4).

theta = Phi (T^T Phi)^-1 and <Psi_T|Phi> are formed once per step from the new determinants and kept; P, Q of the
Hartree energy are kept for the force bias of the next step. Re-orthogonalisation (theta unchanged, overlap divided
by det(R)^mult) and population control (permutation) carry them along; any new determinant clears them.
Checked against fresh evaluations in double precision, closed shell and spin-unrestricted with N_up != N_dn, numpy
and JAX (one LU for theta and the determinant).
Data: $AFQMC_DATA/test/diamond_kpoint (skipped if missing).
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import functions as new
from test_compact_hamiltonian import COMPACT, NE, NG, rel
from test_spin_unrestricted import closed_shell, unrestricted

pytestmark = pytest.mark.skipif(not os.path.isdir(COMPACT), reason="k-point test data not found (set AFQMC_DATA)")


def setups():
    cfg, H, trial, w = closed_shell(6)
    yield "closed", cfg, H, trial, w
    cfg, H, trial, w, _ = unrestricted(6, ne_dn=NE - 1)
    yield "spin", cfg, H, trial, w


def fresh(cfg, trial, w):
    th = [new.biorthogonalize(cfg.backend, t, d) for t, d in zip(new.as_list(trial), w.dets)]
    return th, new.overlap_trial(cfg, trial, w.dets)


def check_kept(cfg, H, trial, w, with_pq):
    th, ov = fresh(cfg, trial, w)
    assert w.theta is not None and w.overlap is not None
    for a, b in zip(w.theta, th):
        assert rel(a, b) < 1e-11
    assert rel(w.overlap, ov) < 1e-11
    if with_pq:
        for (p, q), ch, t in zip(w.pq, H._chans, th):
            p0, q0 = ch.k["pq"](ch.M, t)
            assert rel(p, p0) < 1e-11 and rel(q, q0) < 1e-11


@pytest.mark.parametrize("case", ["closed", "spin"])
def test_kept_quantities_follow_the_walkers(case):
    name, cfg, H, trial, w = next(s for s in setups() if s[0] == case)
    e0 = new.measure_energy(cfg, trial, w, H)[0]
    check_kept(cfg, H, trial, w, with_pq=True)
    h0 = np.exp(cfg.timestep * H.H_zero)
    rng = np.random.default_rng(7)
    for step in range(4):
        w, _ = new.propagate_walkers(cfg, trial, w, H, h0, e0)
        assert w.pq is None
        check_kept(cfg, H, trial, w, with_pq=False)
        if step % 2 == 0:
            e0 = new.measure_energy(cfg, trial, w, H)[0]
            check_kept(cfg, H, trial, w, with_pq=True)
        new.reortho_walkers(cfg, w, "QR")
        check_kept(cfg, H, trial, w, with_pq=step % 2 == 0)
        w.take(rng.integers(0, cfg.num_walkers, cfg.num_walkers))
        check_kept(cfg, H, trial, w, with_pq=step % 2 == 0)
    w.slater_det = w.slater_det + 0.0
    assert w.theta is None and w.overlap is None and w.pq is None


@pytest.mark.parametrize("case", ["closed", "spin"])
def test_step_with_kept_quantities_equals_fresh_step(case):
    """The same step from kept theta, overlap and P, Q and from walkers without them (all evaluated again)."""
    name, cfg, H, trial, w = next(s for s in setups() if s[0] == case)
    e0 = new.measure_energy(cfg, trial, w, H)[0]
    new.reortho_walkers(cfg, w, "QR")
    bare = new.Walkers(slater_det=w.slater_det.copy(), weights=w.weights.copy(),
                       slater_det_dn=None if w.slater_det_dn is None else w.slater_det_dn.copy())
    assert bare.theta is None and w.pq is not None
    H.test_random_field = np.random.default_rng(3).standard_normal((2 * NG, cfg.num_walkers))
    h0 = np.exp(cfg.timestep * H.H_zero)
    n0 = H.force_bias_pq_evaluations
    a, _ = new.propagate_walkers(cfg, trial, w, H, h0, e0)
    assert H.force_bias_pq_evaluations == n0                     # P, Q taken from the energy
    b, _ = new.propagate_walkers(cfg, trial, bare, H, h0, e0)
    assert H.force_bias_pq_evaluations == n0 + 1
    for da, db in zip(a.dets, b.dets):
        assert rel(da, db) < 1e-11
    assert rel(a.weights, b.weights) < 1e-11


def test_jax_theta_and_determinant_from_one_lu():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    be = new.JaxBackend(1, double=True)
    rng = np.random.default_rng(2)
    nbk, nek, nw = 24, 7, 5
    trial = np.eye(nbk, nek)
    dets = rng.standard_normal((nw, nbk, nek)) + 1j * rng.standard_normal((nw, nbk, nek))
    th, det = new.theta_det(be, be.array(trial), be.array(dets))
    th0 = new.biorthogonalize(np, trial, dets)
    det0 = np.linalg.det(trial.T @ dets)
    assert rel(np.asarray(th), th0) < 1e-12 and rel(np.asarray(det), det0) < 1e-12
    # determinant sign with row swaps: permute the occupied rows
    perm = dets[:, [1, 0, 2, 3, 4, 5, 6] + list(range(nek, nbk)), :]
    _, detp = new.theta_det(be, be.array(trial), be.array(perm))
    assert rel(np.asarray(detp), np.linalg.det(trial.T @ perm)) < 1e-12
    q, det_r = new.reortho_with_det(type("C", (), {"backend": be})(), be.array(dets), "Cholesky")
    assert rel(np.linalg.det(trial.T @ np.asarray(q)) * np.asarray(det_r), det0) < 1e-11
