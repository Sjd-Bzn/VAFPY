"""Spin-unrestricted HamiltonianCompact and propagation (vafpy_v4).

1. Identical spins: the unrestricted code with H_up = H_dn and Phi_up = Phi_dn must reproduce the closed-shell code
   (energy, force bias, field, one step and a short trajectory with the same random numbers).
2. Different spin Hamiltonians with a known answer: H_dn is H_up in a basis rotated within the occupied and within
   the virtual orbitals of every k-point, and Phi_dn = U^+ Phi_up. This is the same physical state, so energies and a
   trajectory must agree with the closed-shell run, although every spin-down matrix differs from the spin-up one.
3. N_up != N_dn: the local energy is compared with an independent dense evaluation,
   E = sum_s Tr(h1_s G_s) + [1/2 sum_g rho_g rho_g^+ - 1/2 sum_s sum_g Tr(L_s,g G_s L_s,g^+ G_s)] / nk,
   rho_g = sum_s Tr(L_s,g G_s), rho_g^+ = sum_s Tr(L_s,g^+ G_s), G_s = theta_s T_s^T.
4. Joint compression of two spins keeps sum_g L_s L_s'^+ for every spin pair.
Data: $AFQMC_DATA/test/diamond_kpoint (skipped if missing).
"""
import os
import sys
import numpy as np
import pytest
from mpi4py import MPI
from scipy.linalg import block_diag

sys.path.insert(0, os.path.dirname(__file__))
import functions as new
import compress_H2_compact as comp
from test_compact_hamiltonian import COMPACT, DENSE, DTAU, NB, NE, NG, NK, rel

pytestmark = pytest.mark.skipif(not os.path.isdir(COMPACT), reason="k-point test data not found (set AFQMC_DATA)")


def config(nw, spin=False, ne_dn=None, seed=5, precision="Double"):
    return new.Configuration(
        num_walkers=nw, num_kpoint=NK, num_orbital=NB, num_electron=NE, num_g=NG, singularity=0.0,
        propagator="S2", order_propagation=6, timestep=DTAU, comm=MPI.COMM_WORLD, precision=precision,
        backend=new.NumpyBackend(seed), spin_polarized=spin, num_electron_dn=ne_dn)


def data(cfg):
    q = new.obtain_Q_list(cfg, os.path.join(DENSE, "Q_list.npy"))
    h1 = new.obtain_H1(cfg, os.path.join(DENSE, "H1_svd.npy"))
    h2, layout = new.obtain_H2_host(cfg, os.path.join(COMPACT, "H2_zip.npy"))
    sizes = new.obtain_Q_sizes(cfg, os.path.join(COMPACT, "Q_sizes.npy"))
    return q, h1, h2, layout, sizes


def rotation(seed=11, ne=NE):
    """Block-diagonal unitary, per k-point diag(U_occ, U_virt): the occupied space of the HF trial is kept."""
    rng = np.random.default_rng(seed)

    def unitary(n):
        z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        qm, r = np.linalg.qr(z)
        return qm * (np.diag(r) / abs(np.diag(r)))
    return block_diag(*[block_diag(unitary(ne), unitary(NB - ne)) for _ in range(NK)])


def rotate_compact(h2c, U, kmap, sizes):
    """L'_g = U^+ L_g U for the dense L of every column g, back in the compact layout."""
    Ld = new.compact_to_dense(h2c, kmap, sizes, NB)
    Lr = np.einsum("ap,pqg,qb->abg", U.conj().T, Ld, U, optimize=True)
    return new.dense_to_compact(Lr, kmap, sizes, NB)


def perturb(dets, seed=3, amp=0.05):
    rng = np.random.default_rng(seed)
    return [d + amp * (rng.standard_normal(d.shape) + 1j * rng.standard_normal(d.shape)) for d in dets]


def closed_shell(nw, seed=5):
    cfg = config(nw, seed=seed)
    q, h1, h2, layout, sizes = data(cfg)
    H = new.build_hamiltonian(cfg, h1, h2, layout, q, sizes)
    trial, w = new.initialize_determinant(cfg)
    H.setup_energy_expressions(cfg, trial)
    w.slater_det = perturb([w.slater_det])[0]
    return cfg, H, trial, w


def unrestricted(nw, U=None, seed=5, ne_dn=NE, h_dn=None):
    """Spin-unrestricted set-up; spin down from the rotation U of spin up (U=None: identical spins) or from h_dn."""
    cfg = config(nw, spin=True, ne_dn=ne_dn, seed=seed)
    q, h1, h2, layout, sizes = data(cfg)
    kmap = new.build_kmap(q, NK)
    if h_dn is not None:
        h1_dn, h2_dn = h_dn
    elif U is None:
        h1_dn, h2_dn = h1, h2.copy()
    else:
        h1_dn, h2_dn = U.conj().T @ np.asarray(h1) @ U, rotate_compact(h2, U, kmap, sizes)
    H = new.build_hamiltonian(cfg, h1, h2, layout, q, sizes, one_body_dn=h1_dn, h2_host_dn=h2_dn)
    trial, w = new.initialize_determinant(cfg)
    H.setup_energy_expressions(cfg, trial)
    up = perturb([w.slater_det])[0]
    dn = up if U is None else np.einsum("ap,wpi->wai", U.conj().T, up)
    if ne_dn != NE:
        dn = perturb([w.slater_det_dn], seed=4)[0]
    w.dets = [up, dn.copy()]
    return cfg, H, trial, w, (h1_dn, h2_dn, kmap, sizes)


def trajectory(cfg, H, trial, w, steps):
    e0 = new.measure_energy(cfg, trial, w, H)[0]
    h0 = np.exp(DTAU * H.H_zero)
    out = []
    for _ in range(steps):
        w, _ = new.propagate_walkers(cfg, trial, w, H, h0, e0)
        new.reortho_walkers(cfg, w, "QR")
        e0 = new.measure_energy(cfg, trial, w, H)[0]
        out.append(e0)
    return np.array(out), w


def test_channels():
    assert config(2).channels == [(NE, 2)]
    assert config(2, spin=True).channels == [(NE, 1), (NE, 1)]
    assert config(2, spin=True, ne_dn=3).channels == [(NE, 1), (3, 1)]
    assert config(2, spin=True, ne_dn=0).channels == [(NE, 1)]


@pytest.mark.parametrize("rotated", [False, True])
def test_unrestricted_reproduces_closed_shell(rotated):
    nw = 6
    cfg_c, Hc, tc, wc = closed_shell(nw)
    U = rotation() if rotated else None
    cfg_u, Hu, tu, wu, _ = unrestricted(nw, U)
    assert abs(Hu.H_zero - Hc.H_zero) < 1e-12 * abs(Hc.H_zero)
    if rotated:
        assert rel(Hu._chans[1].h1, Hc.h1) > 1e-2           # the spin-down matrices really differ
    ec = new.measure_components(cfg_c, tc, wc, Hc)
    eu = new.measure_components(cfg_u, tu, wu, Hu)
    for a, b in zip(eu, ec):
        assert abs(a - b) < 1e-11 * abs(b)
    thc = new.walker_thetas(cfg_c, tc, wc)
    thu = new.walker_thetas(cfg_u, tu, wu)
    assert rel(Hu._force_bias(thu), Hc._force_bias(thc)) < 1e-11
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2 * NG, nw))
    Hc.test_random_field = x
    Hu.test_random_field = x
    tr_c, wc2 = trajectory(cfg_c, Hc, tc, wc, 4)
    tr_u, wu2 = trajectory(cfg_u, Hu, tu, wu, 4)
    assert rel(tr_u, tr_c) < 1e-10
    assert rel(wu2.weights, wc2.weights) < 1e-10


def test_unequal_spin_occupation_against_dense_reference():
    nw, ne_dn = 5, 3
    U = rotation(seed=13, ne=ne_dn)
    cfg0 = config(nw, spin=True)
    q, h1, h2, _, sizes = data(cfg0)
    kmap = new.build_kmap(q, NK)
    h_dn = (U.conj().T @ np.asarray(h1) @ U, rotate_compact(h2, U, kmap, sizes))
    cfg, H, trial, w, _ = unrestricted(nw, ne_dn=ne_dn, h_dn=h_dn)
    e = new.measure_energy(cfg, trial, w, H)[0]
    # independent dense evaluation (per walker, then weighted like measure_energy)
    Ls = [new.compact_to_dense(np.asarray(h2), kmap, sizes, NB), new.compact_to_dense(h_dn[1], kmap, sizes, NB)]
    h1s = [np.asarray(h1), h_dn[0]]
    ref = np.zeros(nw, dtype=complex)
    for k in range(nw):
        Gs = []
        for t, d in zip(trial, w.dets):
            th = d[k] @ np.linalg.inv(t.T @ d[k])
            Gs.append(th @ t.T)
        e1 = sum(np.trace(h @ G) for h, G in zip(h1s, Gs))
        rho = sum(np.einsum("pqg,qp->g", L, G) for L, G in zip(Ls, Gs))
        rho_d = sum(np.einsum("qpg,qp->g", L.conj(), G) for L, G in zip(Ls, Gs))
        ex = sum(np.einsum("pqg,qr,srg,sp->", L, G, L.conj(), G, optimize=True) for L, G in zip(Ls, Gs))
        ref[k] = e1 + (0.5 * (rho * rho_d).sum() - 0.5 * ex) / NK
    wts = np.asarray(w.weights)
    e_ref = (ref @ wts) / wts.sum()
    assert abs(e - e_ref) < 1e-11 * abs(e_ref)
    # the closed-shell formula would be wrong here: the check is sensitive to the spin-down occupation
    cfg_c, Hc, tc, wc = closed_shell(nw)
    assert abs(new.measure_energy(cfg_c, tc, wc, Hc)[0] - e_ref) > 1e-3 * abs(e_ref)


def test_unequal_spin_occupation_propagates():
    """N_up != N_dn: a short trajectory runs with finite energies and positive weights, both determinant shapes kept."""
    nw, ne_dn = 4, 3
    cfg, H, trial, w, _ = unrestricted(nw, rotation(seed=13, ne=ne_dn), ne_dn=ne_dn)
    H.test_random_field = np.random.default_rng(9).standard_normal((2 * NG, nw))
    tr, w2 = trajectory(cfg, H, trial, w, 3)
    assert np.all(np.isfinite(tr))
    assert w2.slater_det.shape == (nw, NB * NK, NE * NK) and w2.slater_det_dn.shape == (nw, NB * NK, ne_dn * NK)
    assert np.all(np.asarray(w2.weights).real > 0)


def test_joint_compression_keeps_all_spin_pairs():
    rng = np.random.default_rng(1)
    nb, nk, ng = 3, 2, 40
    raw = [rng.standard_normal((ng, nb, nb * nk)) + 1j * rng.standard_normal((ng, nb, nb * nk)) for _ in range(2)]
    raw[1] = raw[0] + 0.3 * raw[1]                                      # correlated, as for two spins of one system
    c, *_ = comp.svd_one_sector(raw, nb, nk, threshold=1e-12)
    for s in range(2):
        for t in range(2):
            M_s, M_t = (comp.sector_matrix(r, nb, nk) for r in (raw[s], raw[t]))
            C_s, C_t = (np.ascontiguousarray(x.reshape(nk, nb, nb, -1).transpose(0, 2, 1, 3)).reshape(nk * nb * nb, -1)
                        for x in (c[s], c[t]))
            assert rel(C_s @ C_t.conj().T, M_s @ M_t.conj().T) < 1e-12
    assert c[0].shape == c[1].shape
