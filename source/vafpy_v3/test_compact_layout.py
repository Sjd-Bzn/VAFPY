"""Compact H2 layout: momentum map, sector bookkeeping, dense <-> compact conversion.

Compact layout: Lc[(K2,b2), b1, g] = L[(kmap[q(g),K2], b1), (K2,b2), g], shape (nb*nk, nb, ng).
Data: $AFQMC_DATA/test/diamond_kpoint/primitive_k2x2x2{,_compact} (skipped if missing).
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
PR = os.path.join(_DATA_ROOT, "test", "diamond_kpoint", "primitive_k2x2x2")
PRC = os.path.join(_DATA_ROOT, "test", "diamond_kpoint", "primitive_k2x2x2_compact")
need_data = pytest.mark.skipif(not (os.path.isdir(PR) and os.path.isdir(PRC)),
                               reason="k-point test data not found (set AFQMC_DATA)")


def xor_q_list(nk):
    """Z2^n k-mesh: K1 = K2 xor q (zero-based); rows [K1, K2, Q] one-based."""
    return np.array([[(k2 ^ q) + 1, k2 + 1, q + 1] for q in range(nk) for k2 in range(nk)])


def test_kmap_from_qlist():
    kmap = new.build_kmap(xor_q_list(4), 4)
    assert kmap.shape == (4, 4)
    assert all(sorted(kmap[:, k2]) == [0, 1, 2, 3] for k2 in range(4))   # bijection q <-> K1
    assert np.all(kmap[0] == np.arange(4))                                # q=0 is the diagonal


def test_kmap_rejects_non_momentum_conserving_list():
    with pytest.raises(ValueError):
        new.build_kmap(new.build_default_q_list(8), 8)     # |K1-K2| = Q-1 heuristic
    incomplete = xor_q_list(4)[:-1]
    with pytest.raises(ValueError):
        new.build_kmap(incomplete, 4)


def test_roundtrip_synthetic():
    nk, nb, sizes = 4, 3, np.array([5, 2, 3, 4])
    kmap = new.build_kmap(xor_q_list(nk), nk)
    rng = np.random.default_rng(1)
    dense = np.zeros((nb * nk, nb * nk, sizes.sum()), dtype=np.complex128)
    off = new.sector_offsets(sizes)
    for q in range(nk):
        for K2 in range(nk):
            K1 = kmap[q, K2]
            dense[K1 * nb:(K1 + 1) * nb, K2 * nb:(K2 + 1) * nb, off[q]:off[q + 1]] = \
                rng.standard_normal((nb, nb, sizes[q])) + 1j * rng.standard_normal((nb, nb, sizes[q]))
    comp = new.dense_to_compact(dense, kmap, sizes, nb)
    assert comp.shape == (nb * nk, nb, sizes.sum())
    assert np.array_equal(new.compact_to_dense(comp, kmap, sizes, nb), dense)
    # explicit element check of the definition
    q, K2, b1, b2, g = 2, 3, 1, 2, 1
    assert comp[K2 * nb + b2, b1, off[q] + g] == dense[kmap[q, K2] * nb + b1, K2 * nb + b2, off[q] + g]


def test_dense_to_compact_detects_forbidden_blocks():
    nk, nb, sizes = 2, 2, np.array([1, 1])
    kmap = new.build_kmap(xor_q_list(nk), nk)
    dense = np.zeros((nb * nk, nb * nk, 2), dtype=np.complex128)
    dense[0:nb, nb:2 * nb, 0] = 1.0       # (K1=0, K2=1) is not allowed in sector q=0 (K1 must equal K2)
    with pytest.raises(ValueError):
        new.dense_to_compact(dense, kmap, sizes, nb)


def test_layout_detection_and_sector_sizes():
    assert new.is_compact_shape((64, 8, 100), 8, 8)
    assert not new.is_compact_shape((64, 64, 100), 8, 8)
    assert not new.is_compact_shape((8, 8, 36), 8, 1)           # nk=1: identical, treated as dense
    assert new.sector_sizes_default(48, 8).tolist() == [6] * 8
    with pytest.raises(ValueError):
        new.sector_sizes_default(50, 8)


@need_data
def test_primitive_compact_file_matches_dense_file():
    ql = np.load(os.path.join(PR, "Q_list.npy")).T
    kmap = new.build_kmap(ql, 8)
    sizes = np.load(os.path.join(PR, "Q_sizes.npy"))
    dense = np.load(os.path.join(PR, "H2_zip.npy"))
    comp = np.load(os.path.join(PRC, "H2_zip.npy"))
    assert dense.shape == (64, 64, 1999) and comp.shape == (64, 8, 1999)
    assert sizes.sum() == 1999 and len(sizes) == 8
    assert np.array_equal(new.compact_to_dense(comp, kmap, sizes, 8), dense)
    assert comp.nbytes * 8 == dense.nbytes
