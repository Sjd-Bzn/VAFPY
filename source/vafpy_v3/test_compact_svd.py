"""SVD compression of the compact H2 (compress_H2_compact.py)."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import functions as new
import compress_H2_compact as svd
from test_compact_layout import xor_q_list

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get(
    "AFQMC_DATA", os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "data")))
PR = os.path.join(_DATA_ROOT, "test", "diamond_kpoint", "primitive_k2x2x2")


def make_raw(nk, nb, ng, ranks, seed=0):
    """Raw reduced-exporter tensor (ng*nk, nb, nb*nk) whose sector q has rank ranks[q]."""
    rng = np.random.default_rng(seed)
    raw = np.zeros((ng * nk, nb, nb * nk), dtype=np.complex128)
    for q in range(nk):
        rows = nk * nb * nb
        a = rng.standard_normal((rows, ranks[q])) + 1j * rng.standard_normal((rows, ranks[q]))
        b = rng.standard_normal((ranks[q], ng)) + 1j * rng.standard_normal((ranks[q], ng))
        M = a @ b                                          # [(K2,b1,b2), g]
        x = M.reshape(nk, nb, nb, ng).transpose(3, 1, 0, 2)  # g, b1, K2, b2
        raw[q * ng:(q + 1) * ng] = x.reshape(ng, nb, nk * nb)
    return raw


def test_sector_matrix_inverts_layout():
    nk, nb, ng = 4, 2, 5
    raw = make_raw(nk, nb, ng, [2] * nk)
    M = svd.sector_matrix(raw[:ng], nb, nk)
    # M[(K2,b1,b2), g] == raw[g, b1, K2*nb + b2]
    for K2, b1, b2, g in [(0, 0, 0, 0), (3, 1, 0, 4), (2, 1, 1, 2)]:
        assert M[(K2 * nb + b1) * nb + b2, g] == raw[g, b1, K2 * nb + b2]


def test_compress_ranks_shape_and_gram(tmp_path):
    nk, nb, ng = 4, 2, 40
    ranks = [3, 5, 5, 4]
    raw = make_raw(nk, nb, ng, ranks)
    np.save(tmp_path / "H2.npy", raw)
    np.save(tmp_path / "Q_list.npy", xor_q_list(nk).T)
    h2_zip, sizes = svd.compress(str(tmp_path / "H2.npy"), str(tmp_path / "Q_list.npy"),
                                 nb, nk, 1e-8, 1)
    assert sizes.tolist() == ranks
    assert h2_zip.shape == (nk * nb, nb, sum(ranks))
    kmap = new.build_kmap(xor_q_list(nk), nk)
    dense = new.compact_to_dense(h2_zip, kmap, sizes, nb)         # (nb nk, nb nk, ng_kept)
    # sum_g L[pr,g] L*[qs,g] must equal the raw one, with the raw tensor expanded the same way
    raw_dense = np.zeros((nb * nk, nb * nk, raw.shape[0]), dtype=np.complex128)
    for q in range(nk):
        for K2 in range(nk):
            K1 = kmap[q, K2]
            blk = raw[q * ng:(q + 1) * ng, :, K2 * nb:(K2 + 1) * nb]      # g, b1, b2
            raw_dense[K1 * nb:(K1 + 1) * nb, K2 * nb:(K2 + 1) * nb, q * ng:(q + 1) * ng] = blk.transpose(1, 2, 0)
    gram_raw = raw_dense.reshape(-1, raw_dense.shape[2]) @ raw_dense.reshape(-1, raw_dense.shape[2]).conj().T
    gram_new = dense.reshape(-1, dense.shape[2]) @ dense.reshape(-1, dense.shape[2]).conj().T
    assert np.abs(gram_raw - gram_new).max() < 1e-9 * np.abs(gram_raw).max()


def test_threshold_truncates(tmp_path):
    nk, nb, ng = 2, 2, 30
    raw = make_raw(nk, nb, ng, [6, 6])
    np.save(tmp_path / "H2.npy", raw)
    np.save(tmp_path / "Q_list.npy", xor_q_list(nk).T)
    _, s_all = svd.compress(str(tmp_path / "H2.npy"), str(tmp_path / "Q_list.npy"), nb, nk, 1e-12, 1)
    # a threshold above all singular values keeps one (zero) column per sector, never an empty tensor
    h2_zip, s_none = svd.compress(str(tmp_path / "H2.npy"), str(tmp_path / "Q_list.npy"), nb, nk, 1e9, 1)
    assert s_all.tolist() == [6, 6]
    assert s_none.tolist() == [1, 1] and np.all(h2_zip == 0)


def test_rejects_wrong_layout(tmp_path):
    np.save(tmp_path / "H2.npy", np.zeros((40, 4, 4), dtype=np.complex128))   # not (ng*nk, nb, nb*nk)
    np.save(tmp_path / "Q_list.npy", xor_q_list(2).T)
    with pytest.raises(SystemExit):
        svd.compress(str(tmp_path / "H2.npy"), str(tmp_path / "Q_list.npy"), 2, 2, 1e-4, 1)
