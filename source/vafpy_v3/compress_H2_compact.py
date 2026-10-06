#!/usr/bin/env python3
"""compress_H2_compact.py - SVD-compress the compact H2 per momentum sector for vafpy_v3.

Input  (from the reduced VASP exporter, numpy order):
    H2.npy        (ng*nk, nb, nb*nk)   index [(q,G), b1, (K2,b2)]
    H1.npy        (nb, nb, nk)
    Q_list.npy    rows [K1, K2, Q] (one-based), shape (3, nk^2) or (nk^2, 3)
Output:
    H2_zip.npy    (nb*nk, nb, ng_kept)  compact layout  Lc[(K2,b2), b1, g]
    Q_sizes.npy   (nk,)                 columns kept per q sector (sum = NGVEC)
    H1_svd.npy    (nb, nb, nk)

Per sector q the rows (K2, b1, b2) form the matrix M_q (nk*nb*nb, ng); the pair
(K1, K2) of every block is fixed by momentum conservation, K1 = kmap[q, K2].
The SVD is M_q = U S V^+ and the columns with s > threshold are kept as U S, so
that sum_g L L^+ = M M^+ is preserved up to the discarded singular values (same
convention and threshold as compress_H2.py for the full layout).

Usage:
    python3 compress_H2_compact.py --h2 H2.npy --h1 H1.npy --qlist Q_list.npy \
            --num-orb 8 --num-k 8 [--threshold 1e-4] [--jobs 8] [--outdir .]
"""
import argparse
import os
import sys
from time import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functions import build_kmap  # noqa: E402


def load_q_list(path):
    ql = np.load(path)
    if ql.shape[0] == 3 and ql.ndim == 2 and ql.shape[1] != 3:
        ql = ql.T
    return ql.astype(np.int64)


def sector_matrix(raw_q, num_orb, num_k):
    """raw_q: (ng, b1, (K2,b2)) -> M[(K2,b1,b2), g]"""
    ng = raw_q.shape[0]
    x = raw_q.reshape(ng, num_orb, num_k, num_orb)       # g, b1, K2, b2
    return np.ascontiguousarray(x.transpose(2, 1, 3, 0)).reshape(num_k * num_orb * num_orb, ng)


def svd_one_sector(raw_q, num_orb, num_k, threshold):
    M = sector_matrix(raw_q, num_orb, num_k)
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    kept = s > threshold
    n_kept = int(kept.sum())
    if n_kept == 0:                                       # keep one (zero) column so shapes stay valid
        comp = np.zeros((M.shape[0], 1), dtype=np.complex128)
        n_kept = 1
        s_min = 0.0
    else:
        comp = u[:, kept] * s[kept]                       # U S
        s_min = float(s[kept][-1])
    gram = M @ M.conj().T
    dgram = gram - comp @ comp.conj().T
    err_rel = float(np.linalg.norm(dgram) / max(np.linalg.norm(gram), 1e-300))
    err_max = float(np.abs(dgram).max())
    # back to compact layout [(K2,b2), b1, n]
    c = comp.reshape(num_k, num_orb, num_orb, n_kept)     # K2, b1, b2, n
    c = np.ascontiguousarray(c.transpose(0, 2, 1, 3)).reshape(num_k * num_orb, num_orb, n_kept)
    return c, float(s[0]), s_min, n_kept, err_rel, err_max, M.shape


def compress(h2_path, qlist_path, num_orb, num_k, threshold, n_jobs):
    t0 = time()
    raw = np.load(h2_path)
    print(f"Loading H2 from  : {h2_path}\n  raw shape      : {raw.shape} dtype={raw.dtype} ({raw.nbytes / 2**20:.0f} MB)")
    nb_tot = num_orb * num_k
    if raw.ndim != 3 or raw.shape[1] != num_orb or raw.shape[2] != nb_tot:
        sys.exit(f"ERROR: expected compact H2 (ng*nk, {num_orb}, {nb_tot}), got {raw.shape}")
    if raw.shape[0] % num_k:
        sys.exit(f"ERROR: first axis {raw.shape[0]} is not a multiple of num_k={num_k}")
    ng_raw = raw.shape[0] // num_k
    raw = raw.astype(np.complex128, copy=False)
    kmap = build_kmap(load_q_list(qlist_path), num_k)     # validates momentum conservation
    print(f"  ng per q (raw) : {ng_raw}   sectors: {num_k}   kmap checked ({qlist_path})")
    print(f"\nSVD per momentum sector (threshold={threshold}, n_jobs={n_jobs})")
    print(f"{'Q':>3} {'matrix':>14} {'kept':>6} {'s_max':>10} {'s_min_kept':>11} {'rel.err(LL+)':>13} {'max.err':>9}")
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        res = list(pool.map(
            lambda q: svd_one_sector(raw[q * ng_raw:(q + 1) * ng_raw], num_orb, num_k, threshold),
            range(num_k)))
    parts, sizes = [], []
    for q, (c, smax, smin, nk_, er, em, shp) in enumerate(res, start=1):
        parts.append(c)
        sizes.append(nk_)
        print(f"{q:>3} {str(shp):>14} {nk_:>6} {smax:>10.4f} {smin:>11.2e} {er:>13.2e} {em:>9.1e}")
    h2_zip = np.concatenate(parts, axis=2).astype(np.complex128)
    sizes = np.array(sizes, dtype=np.int64)
    print("\nCompression summary")
    print(f"  Input  (ng*nk, nb, nb*nk)        : {raw.shape}")
    print(f"  Output (nb*nk, nb, ng_kept)      : {h2_zip.shape}   ng_kept = {sizes.sum()}  per q: {sizes.tolist()}")
    print(f"  Columns kept / raw (per sector)  : {sizes.sum()} / {raw.shape[0]}  ({raw.shape[0] / sizes.sum():.1f}x)")
    print(f"  Memory raw -> compressed         : {raw.nbytes / 2**20:.0f} MB -> {h2_zip.nbytes / 2**20:.1f} MB")
    print(f"  (total runtime: {time() - t0:.1f} s)")
    return h2_zip, sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2", required=True)
    ap.add_argument("--h1", required=True)
    ap.add_argument("--qlist", required=True)
    ap.add_argument("--num-orb", type=int, required=True)
    ap.add_argument("--num-k", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    h2_zip, sizes = compress(a.h2, a.qlist, a.num_orb, a.num_k, a.threshold, a.jobs)
    os.makedirs(a.outdir, exist_ok=True)
    np.save(os.path.join(a.outdir, "H2_zip.npy"), h2_zip)
    np.save(os.path.join(a.outdir, "Q_sizes.npy"), sizes)
    h1 = np.load(a.h1).astype(np.complex128)
    np.save(os.path.join(a.outdir, "H1_svd.npy"), h1)
    print(f"\nSaved to {a.outdir}: H2_zip.npy {h2_zip.shape}, Q_sizes.npy, H1_svd.npy {h1.shape}")
    print(f"Set in vafpy.in:  NGVEC : {sizes.sum()}   KPOINT : {a.num_k}   NORB : {a.num_orb}")


if __name__ == "__main__":
    main()
