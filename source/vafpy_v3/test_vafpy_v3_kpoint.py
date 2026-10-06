"""k-point validation of vafpy_v3: primitive cell + 2x2x2 k-mesh vs equivalent 2x2x2 supercell at Gamma.

Both describe the same finite system and the same 64-orbital space (8 HF orbitals/k vs 64 supercell
orbitals), so HF energy, MP2 energy and the imaginary-time dynamics must agree. These tests encode
the checks that were used to validate the k-point path (H_zero, 1/nk interaction normalisation,
single-precision rebalance).

Data (large, not in git): $AFQMC_DATA/test/diamond_kpoint/{supercell_gamma,primitive_k2x2x2}
(see README.txt there). Tests are skipped if the data is missing.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import functions as new

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get(
    "AFQMC_DATA", os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "data")))
KDIR = os.path.join(_DATA_ROOT, "test", "diamond_kpoint")
SC = os.path.join(KDIR, "supercell_gamma")
PR = os.path.join(KDIR, "primitive_k2x2x2")
PRC = os.path.join(KDIR, "primitive_k2x2x2_compact")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(SC) and os.path.isdir(PR) and os.path.isdir(PRC)),
    reason="k-point test data not found (set AFQMC_DATA)")

# system definitions: (dir, num_k, orbitals per k, electrons per k, num_g)
SYSTEMS = {
    "supercell": dict(dir=SC, num_k=1, num_orb=64, num_e=32, num_g=1769),
    "primitive": dict(dir=PR, num_k=8, num_orb=8, num_e=4, num_g=1999),                 # dense H2
    "primitive_compact": dict(dir=PRC, num_k=8, num_orb=8, num_e=4, num_g=1999),         # compact H2
}
K_SYSTEMS = ("primitive", "primitive_compact")
DTAU = 2.5e-4


def build(name, num_walkers=4, seed=12345, precision="Double"):
    s = SYSTEMS[name]
    from mpi4py import MPI
    backend = new.NumpyBackend(seed=seed)
    config = new.Configuration(
        num_walkers=num_walkers, num_kpoint=s["num_k"], num_orbital=s["num_orb"],
        num_electron=s["num_e"], num_g=s["num_g"], singularity=0.0, propagator="S2",
        order_propagation=6, timestep=DTAU, comm=MPI.COMM_WORLD, precision=precision, backend=backend)
    qfile = os.path.join(s["dir"], "Q_list.npy")
    h2, layout = new.obtain_H2_host(config, os.path.join(s["dir"], "H2_zip.npy"))
    sizes = new.obtain_Q_sizes(config, os.path.join(s["dir"], "Q_sizes.npy")) if layout == "compact" else None
    H = new.build_hamiltonian(
        config, new.obtain_H1(config, os.path.join(s["dir"], "H1_svd.npy")), h2, layout,
        new.obtain_Q_list(config, qfile), sizes)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    return config, H, trial, walkers


def test_hf_energy_primitive_equals_supercell():
    """HF energy and its components (E1, Hartree, exchange) are the same 8-cell totals in both setups."""
    res = {}
    for name in SYSTEMS:
        config, H, trial, walkers = build(name)
        e1, eh, ex = new.measure_components(config, trial, walkers, H)
        e_tot = new.measure_energy(config, trial, walkers, H)[0]
        res[name] = np.array([e_tot.real, e1.real, eh.real, ex.real])
    for name in K_SYSTEMS:
        assert np.allclose(res[name], res["supercell"], atol=1e-3), (name, res)
    assert abs(res["supercell"][0] - 455.795) < 1e-2, res["supercell"]


def test_h_zero_scales_determinant_as_hartree_energy():
    """exp(dtau*H_zero) applied to every column of the full determinant must give exp(dtau*E_H_total):
    H_zero = E_H_total / (2 * ne * nk)."""
    for name, s in SYSTEMS.items():
        config, H, trial, walkers = build(name)
        e_h_total = new.measure_components(config, trial, walkers, H)[1].real
        expected = e_h_total / (2 * s["num_e"] * s["num_k"])
        assert abs(H.H_zero.real - expected) < 1e-6 * abs(expected), (name, H.H_zero, expected)


def _mp2(name, scale):
    """MP2 from the exported (compressed) L in the HF-eigenstate basis; V(ia|jb) = scale * sum_G L_ia conj(L_jb)."""
    s = SYSTEMS[name]
    nk, no, nv = s["num_k"], s["num_e"], s["num_orb"] - s["num_e"]
    L = np.load(os.path.join(s["dir"], "H2_zip.npy"))                    # (nb*nk, nb*nk, G) or compact
    if new.is_compact_shape(L.shape, s["num_orb"], nk):
        kmap = new.build_kmap(np.load(os.path.join(s["dir"], "Q_list.npy")).T, nk)
        L = new.compact_to_dense(L, kmap, np.load(os.path.join(s["dir"], "Q_sizes.npy")), s["num_orb"])
    eps = np.load(os.path.join(s["dir"], "eigenvalues.npy"))[:, :, 0].T.ravel()   # index (k*nb + band)
    occ = np.array([k * s["num_orb"] + b for k in range(nk) for b in range(no)])
    vir = np.array([k * s["num_orb"] + b for k in range(nk) for b in range(no, s["num_orb"])])
    Lov = L[np.ix_(occ, vir)].reshape(len(occ) * len(vir), -1)           # ((i,a), G)
    M = (Lov @ Lov.conj().T * scale).reshape(len(occ), len(vir), len(occ), len(vir))
    ex = M.transpose(0, 3, 2, 1)
    eo, ev = eps[occ], eps[vir]
    den = eo[:, None, None, None] + eo[None, None, :, None] - ev[None, :, None, None] - ev[None, None, None, :]
    return (M * (2 * M - ex) / den).sum().real


def test_mp2_primitive_equals_supercell_with_1_over_nk():
    """The exported L is unscaled: the physical interaction is (1/nk) sum L L^+.
    MP2 in the identical orbital space must agree between supercell (nk=1) and primitive (nk=8, scale 1/8)."""
    mp2_sc = _mp2("supercell", 1.0)
    for name in K_SYSTEMS:
        mp2_pr = _mp2(name, 1.0 / SYSTEMS[name]["num_k"])
        assert abs(mp2_sc - mp2_pr) < 1e-3, (name, mp2_sc, mp2_pr)
        # sanity: without the 1/nk factor the energies would be wildly different
        assert abs(_mp2(name, 1.0) - mp2_sc) > 100
    assert abs(mp2_sc - (-18.4343)) < 5e-3, mp2_sc


def _decay(name, num_walkers, nsteps, seed):
    config, H, trial, walkers = build(name, num_walkers=num_walkers, seed=seed)
    e_hf = new.measure_energy(config, trial, walkers, H)[0].real
    h_0 = np.exp(config.timestep * H.H_zero)
    e0 = e_hf
    mean_w = []
    for _ in range(nsteps):
        walkers, nrare = new.propagate_walkers(config, trial, walkers, H, h_0, e0)
        walkers.slater_det = new.reortho_qr(config, walkers.slater_det)
        e0 = new.measure_energy(config, trial, walkers, H)[0]
        mean_w.append(np.mean(walkers.weights).real)
        assert nrare == 0
    assert np.isfinite(e0) and np.all(np.isfinite(walkers.weights))
    return e_hf - e0.real, np.array(mean_w)


def test_short_time_energy_decay_primitive_equals_supercell():
    """Early imaginary-time decay of the mixed energy is -tau*Var(E_L) and must match between the two
    descriptions; walker weights stay O(1). With the interaction mis-scaled by nk the primitive decay is ~nk x too fast,
    and a mis-scaled H_zero zeroes all weights."""
    nw, nsteps = 128, 8
    d_sc, w_sc = _decay("supercell", nw, nsteps, seed=1)
    assert np.all(w_sc > 0.9) and np.all(w_sc < 1.1), w_sc
    for name in K_SYSTEMS:
        d_pr, w_pr = _decay(name, nw, nsteps, seed=2)
        ratio = d_pr / d_sc
        assert 0.8 < ratio < 1.25, (name, d_sc, d_pr, ratio)
        assert np.all(w_pr > 0.9) and np.all(w_pr < 1.1), (name, w_pr)


def test_rebalance_global_single_precision():
    """rebalance_global must accept complex64 walkers/weights (PRECSION: Single) without corrupting them."""
    from mpi4py import MPI
    config, H, trial, walkers = build("primitive", num_walkers=8, precision="Single")
    rng = np.random.default_rng(0)
    mats = (rng.standard_normal((8, 32, 32)) + 1j * rng.standard_normal((8, 32, 32))).astype(np.complex64)
    wts = (rng.uniform(0.5, 1.5, 8)).astype(np.complex64)
    new_mats, new_w = new.rebalance_global(MPI.COMM_WORLD, mats, wts, config)
    assert np.all(np.isfinite(new_mats)) and np.all(np.isfinite(new_w))
    assert np.allclose(new_w, 1.0)
    # every resampled walker must be one of the originals
    for m in new_mats:
        assert any(np.allclose(m, o, atol=1e-6) for o in mats)
