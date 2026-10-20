"""3x3x1 k-mesh (Z3 x Z3 group): primitive cell with 9 k-points against the equivalent 3x3x1 supercell at Gamma.

Unlike the 2x2x2 mesh, kmap[q,K2] is not symmetric in (q,K2) and q != -q, so orientation mistakes (which row/column
carries the implicit k point, which way the second pair of an integral is ordered) are visible here.
Both describe the same 72-orbital HF space (8 lowest HF bands per k = 72 lowest supercell HF orbitals).
Data: $AFQMC_DATA/test/diamond_kpoint331/{primitive_dense,primitive_compact,supercell} (skipped if missing).
"""
import os
import sys
import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(__file__))
import functions as new

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get(
    "AFQMC_DATA", os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "data")))
KD = os.path.join(_DATA_ROOT, "test", "diamond_kpoint331")
PD, PC, SC = (os.path.join(KD, d) for d in ("primitive_dense", "primitive_compact", "supercell"))
pytestmark = pytest.mark.skipif(not all(os.path.isdir(d) for d in (PD, PC, SC)),
                                reason="3x3x1 test data not found (set AFQMC_DATA)")

NK, NB, NE, NG = 9, 8, 4, 2162
DTAU = 2.5e-4
VASP_TOTAL = 743.72153639287 + 260.95171318919 - 590.69835364602          # E1 + EH + EX exported for the primitive cell


def make_config(nw, nk, nb, ne, ng, seed=5):
    return new.Configuration(num_walkers=nw, num_kpoint=nk, num_orbital=nb, num_electron=ne, num_g=ng, singularity=0.0,
                             propagator="S2", order_propagation=6, timestep=DTAU, comm=MPI.COMM_WORLD,
                             precision="Double", backend=new.NumpyBackend(seed))


def primitive(layout, nw=4, seed=5):
    d = PC if layout == "compact" else PD
    c = make_config(nw, NK, NB, NE, NG, seed)
    q = new.obtain_Q_list(c, os.path.join(d, "Q_list.npy"))
    h2, lay = new.obtain_H2_host(c, os.path.join(d, "H2_zip.npy"))
    assert lay == layout
    sizes = new.obtain_Q_sizes(c, os.path.join(d, "Q_sizes.npy")) if lay == "compact" else None
    H = new.build_hamiltonian(c, new.obtain_H1(c, os.path.join(d, "H1_svd.npy")), h2, lay, q, sizes)
    trial, w = new.initialize_determinant(c)
    H.setup_energy_expressions(c, trial)
    return c, H, trial, w


def supercell(nw=4, seed=5):
    ng = np.load(os.path.join(SC, "H2_zip.npy"), mmap_mode="r").shape[2]
    c = make_config(nw, 1, 72, 36, ng, seed)
    h2, lay = new.obtain_H2_host(c, os.path.join(SC, "H2_zip.npy"))
    H = new.build_hamiltonian(c, new.obtain_H1(c, os.path.join(SC, "H1_svd.npy")), h2, lay, None)
    trial, w = new.initialize_determinant(c)
    H.setup_energy_expressions(c, trial)
    return c, H, trial, w


def perturbed(c, w, amp=0.05, seed=3):
    rng = np.random.default_rng(seed)
    sd = np.asarray(w.slater_det)
    sd = sd + amp * (rng.standard_normal(sd.shape) + 1j * rng.standard_normal(sd.shape))
    w.slater_det = c.backend.array(sd, dtype=c.complex_type)
    return w


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def mp2(L, eps, occ, vir, scale):
    """E = sum conj(M)(2M - ex)/D with M[i,a,j,b] = (ia|jb) = scale * sum_g L[i,a] conj(L[b,j])."""
    Lo, Lv = L[np.ix_(occ, vir)], L[np.ix_(vir, occ)].transpose(1, 0, 2)
    M = np.einsum("iag,jbg->iajb", Lo, Lv.conj(), optimize=True) * scale
    ex = M.transpose(0, 3, 2, 1)
    eo, ev = eps[occ], eps[vir]
    D = eo[:, None, None, None] + eo[None, None, :, None] - ev[None, :, None, None] - ev[None, None, None, :]
    return (M.conj() * (2 * M - ex) / D).sum().real


# ---------------------------------------------------------------------------------------------------------
def test_kmap_is_not_symmetric_and_is_a_group_table():
    kmap = new.build_kmap(np.load(os.path.join(PC, "Q_list.npy")).T, NK)
    assert not np.array_equal(kmap, kmap.T)                         # the 2x2x2 mesh would be symmetric
    for K2 in range(NK):
        assert sorted(kmap[:, K2]) == list(range(NK))               # q <-> K1 bijection for every column k point
    assert np.array_equal(kmap[0], np.arange(NK))                   # q = 0 is the diagonal


def test_compact_file_equals_dense_file():
    kmap = new.build_kmap(np.load(os.path.join(PC, "Q_list.npy")).T, NK)
    sizes = np.load(os.path.join(PC, "Q_sizes.npy"))
    dense = np.load(os.path.join(PD, "H2_zip.npy")); comp = np.load(os.path.join(PC, "H2_zip.npy"))
    assert dense.shape == (72, 72, NG) and comp.shape == (72, 8, NG)
    # both SVDs keep the same columns here; the two files describe the same interaction up to a unitary in each sector
    gram = lambda z: z.reshape(-1, z.shape[2]) @ z.reshape(-1, z.shape[2]).conj().T
    assert rel(gram(new.compact_to_dense(comp, kmap, sizes, NB)), gram(dense)) < 1e-8


def test_hf_energy_matches_vasp_and_supercell():
    res = {}
    for name, (c, H, t, w) in (("compact", primitive("compact")), ("dense", primitive("dense")), ("supercell", supercell())):
        e1, eh, ex = new.measure_components(c, t, w, H)
        res[name] = np.array([new.measure_energy(c, t, w, H)[0].real, e1.real, eh.real, ex.real])
    assert abs(res["compact"][0] - VASP_TOTAL) < 1e-4, res
    for name in ("dense", "supercell"):
        assert np.allclose(res[name], res["compact"], atol=1e-3), (name, res)


def test_mp2_primitive_equals_supercell_on_a_non_symmetric_mesh():
    Ls = np.load(os.path.join(SC, "H2_zip.npy")); es = np.load(os.path.join(SC, "eigenvalues.npy"))[:, 0, 0]
    kmap = new.build_kmap(np.load(os.path.join(PC, "Q_list.npy")).T, NK)
    Lp = new.compact_to_dense(np.load(os.path.join(PC, "H2_zip.npy")), kmap, np.load(os.path.join(PC, "Q_sizes.npy")), NB)
    ep = np.load(os.path.join(PC, "eigenvalues.npy"))[:, :, 0].T.ravel()                 # index k*nb + band
    occ = np.array([k * NB + b for k in range(NK) for b in range(NE)])
    vir = np.array([k * NB + b for k in range(NK) for b in range(NE, NB)])
    m_sc = mp2(Ls, es, np.arange(36), np.arange(36, 72), 1.0)
    m_pr = mp2(Lp, ep, occ, vir, 1.0 / NK)
    assert abs(m_sc - (-20.5657)) < 1e-3, m_sc
    assert abs(m_pr - m_sc) < 1e-3, (m_pr, m_sc)
    # the pair-orientation of the exchange partner matters on this mesh: the reversed ordering must NOT agree
    Lo = Lp[np.ix_(occ, vir)]
    wrong = np.einsum("iag,jbg->iajb", Lo, Lo.conj(), optimize=True) / NK
    assert abs(wrong.shape[0] - len(occ)) == 0
    assert abs((np.einsum("iag,jbg->iajb", Lo, Lp[np.ix_(vir, occ)].transpose(1, 0, 2).conj(), optimize=True) / NK - wrong)).max() > 1e-3


def dense_from_compact(nw, seed=5):
    """Dense (v2-style) Hamiltonian built from exactly the same numbers as the compact one. The independently SVD-compressed
    dense file agrees on every rotation-invariant quantity but not column by column, so per-column kernels (force bias,
    auxiliary field) must be compared on a pair that shares its columns."""
    c, Hc, t, w = primitive("compact", nw=nw, seed=seed)
    kmap = new.build_kmap(np.load(os.path.join(PC, "Q_list.npy")).T, NK)
    dense = new.compact_to_dense(np.load(os.path.join(PC, "H2_zip.npy")), kmap, np.load(os.path.join(PC, "Q_sizes.npy")), NB)
    q = new.obtain_Q_list(c, os.path.join(PC, "Q_list.npy"))
    Hd = new.build_hamiltonian(c, new.obtain_H1(c, os.path.join(PC, "H1_svd.npy")), dense, "dense", q)
    Hd.setup_energy_expressions(c, t)
    return c, Hd, Hc, t, w


def test_independently_compressed_files_agree_on_invariants():
    """Dense and compact SVD files (separate compressions) give the same energies for walkers away from the HF point."""
    cd, Hd, td, wd = primitive("dense", nw=5); cc, Hc, tc, wc = primitive("compact", nw=5)
    wd, wc = perturbed(cd, wd), perturbed(cc, wc)
    th = new.biorthogonalize(cd.backend, td, wd.slater_det)
    for name in ("compute_one_body", "compute_hartree", "compute_exchange"):
        assert rel(getattr(Hc, name)(th), getattr(Hd, name)(th)) < 1e-6, name


def test_compact_hamiltonian_equals_dense_at_nk9():
    cd, Hd, Hc, td, wd = dense_from_compact(5)
    cc, tc = cd, td
    wc = new.Walkers(slater_det=wd.slater_det.copy(), weights=wd.weights.copy())
    assert abs(Hd.H_zero - Hc.H_zero) < 1e-10 * abs(Hd.H_zero)
    assert rel(Hc.exp_h1_half, Hd.exp_h1_half) < 1e-10
    wd, wc = perturbed(cd, wd), perturbed(cc, wc)
    th = new.biorthogonalize(cd.backend, td, wd.slater_det)
    for name in ("compute_one_body", "compute_hartree", "compute_exchange"):
        assert rel(getattr(Hc, name)(th), getattr(Hd, name)(th)) < 1e-11, name
    assert rel(Hc._force_bias(th), Hd._force_bias_expression(th)) < 1e-11
    x = np.random.default_rng(4).standard_normal((2 * NG, 5)) + 0.3j * np.random.default_rng(5).standard_normal((2 * NG, 5))
    xb = cd.backend.array(x, dtype=cd.complex_type)
    assert rel(Hc._auxiliary_field(xb), Hd._auxiliary_field(xb)) < 1e-11
    Hc.exchange_mode = "direct"
    try:
        assert rel(Hc.compute_exchange(th), Hd.compute_exchange(th)) < 1e-11
    finally:
        Hc.exchange_mode = "gram"
    f = cd.backend.array(np.random.default_rng(11).standard_normal((2 * NG, 5)), dtype=cd.float_type)
    Hd.test_random_field = Hc.test_random_field = f
    e0 = new.measure_energy(cd, td, wd, Hd)[0]
    nd, _ = new.propagate_walkers(cd, td, wd, Hd, np.exp(DTAU * Hd.H_zero), e0)
    nc, _ = new.propagate_walkers(cc, tc, wc, Hc, np.exp(DTAU * Hc.H_zero), e0)
    assert rel(nc.slater_det, nd.slater_det) < 1e-11 and rel(nc.weights, nd.weights) < 1e-11


def _decay(c, H, t, w, nsteps):
    e_hf = new.measure_energy(c, t, w, H)[0].real
    h0 = np.exp(c.timestep * H.H_zero); e0 = e_hf; mw = []
    for _ in range(nsteps):
        w, nrare = new.propagate_walkers(c, t, w, H, h0, e0)
        w.slater_det = new.reortho_qr(c, w.slater_det)
        e0 = new.measure_energy(c, t, w, H)[0]
        mw.append(np.mean(w.weights).real)
        assert nrare == 0
    assert np.isfinite(e0)
    return e_hf - e0.real, np.array(mw)


def test_short_time_decay_primitive_equals_supercell():
    d_pr, w_pr = _decay(*primitive("compact", nw=96, seed=2), 6)
    d_sc, w_sc = _decay(*supercell(nw=96, seed=1), 6)
    assert 0.75 < d_pr / d_sc < 1.33, (d_pr, d_sc)
    for w in (w_pr, w_sc):
        assert np.all(w > 0.9) and np.all(w < 1.1), w
