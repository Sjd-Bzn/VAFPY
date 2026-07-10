#!/usr/bin/env python3
"""
compress_H2.py — SVD-compress H2.npy per momentum sector for vafpy_v2.

Replaces: afqmc_INPUT_H2_11.py, afqmc_funcs_kpts_11.py, afqmc_alpha_11.py,
          afqmc_RUN_kpts_11.py, svd_kpts.py

Usage (edit the CONFIG block below, then run):
    python3 compress_H2.py

    # control BLAS threads per worker (total cores = N_JOBS * OMP_NUM_THREADS)
    OMP_NUM_THREADS=4 python3 compress_H2.py

Outputs:
    H2_zip.npy  — compressed two-body Hamiltonian, shape (nb*nk, nb*nk, ng_compressed)
    H1_svd.npy  — H1 renamed (shape unchanged)
    Prints NGVEC value to put in vafpy.in.

Input H2.npy expected from VASP: shape (ng_raw, nb*nk, nb*nk).
Q_list.npy: shape (3, N) or (N, 3), columns [K1, K2, Q] (1-indexed).
"""

import numpy as np
import sys
from time import time
from joblib import Parallel, delayed

# =============================================================================
#  CONFIG  — edit these for your system
# =============================================================================
H2_IN      = "../vasp/H2.npy"    # raw H2 from VASP, shape (ng_raw, nb*nk, nb*nk)
H1_IN      = "../vasp/H1.npy"    # H1 from VASP, shape (nb, nb, nk)
# For 2x2x2 diamond (8 k-points) use Q_list_2k.npy from kpts_code/:
#   cp /path/to/kpts_code/Q_list_2k.npy Q_list.npy
QLIST_IN   = "Q_list.npy"         # shape (3, nk^2) or (nk^2, 3), cols [K1, K2, Q]
H2_OUT     = "H2_zip.npy"         # output: shape (nb*nk, nb*nk, ng_compressed)
H1_OUT     = "H1_svd.npy"         # output H1 (shape unchanged, just renamed)

NUM_ORB    = 8      # bands per k-point
NUM_K      = 8      # k-points  (2×2×2 → 8)
SVD_THRESH = 1e-4   # keep singular values s > SVD_THRESH
N_JOBS     = 8      # parallel Q sectors (set to NUM_K or number of cores)
# =============================================================================


# ---------------------------------------------------------------------------
#  Q-list helpers
# ---------------------------------------------------------------------------
def load_q_list(path, num_k):
    """Load Q_list.npy; auto-transpose (3,N) → (N,3). Falls back to heuristic."""
    try:
        ql = np.load(path)
        if ql.shape[0] == 3 and ql.ndim == 2:
            ql = ql.T
        return ql.astype(np.int64)
    except FileNotFoundError:
        print(f"  WARNING: {path} not found — using |K1-K2|==Q-1 heuristic.")
        print("  This is ONLY correct for linearly ordered k-points.")
        rows = []
        for k1 in range(1, num_k + 1):
            for k2 in range(1, num_k + 1):
                for q in range(1, num_k + 1):
                    if abs(k1 - k2) == q - 1:
                        rows.append([k1, k2, q])
        return np.array(rows, dtype=np.int64)


def get_k1s_k2s(q_list, q_selected):
    mask = q_list[:, 2] == q_selected
    sub  = q_list[mask]
    return list(zip(sub[:, 0], sub[:, 1]))


def extract_q_block(h2, q_list, q_selected, num_orb):
    """Return H2 with only the blocks belonging to Q=q_selected (rest zeroed)."""
    result = np.zeros_like(h2)
    for K1, K2 in get_k1s_k2s(q_list, q_selected):
        r1 = (K1 - 1) * num_orb;  r2 = K1 * num_orb
        c1 = (K2 - 1) * num_orb;  c2 = K2 * num_orb
        result[r1:r2, c1:c2, :] = h2[r1:r2, c1:c2, :]
    return result


# ---------------------------------------------------------------------------
#  Per-Q SVD worker (called in parallel)
# ---------------------------------------------------------------------------
def _svd_one_q(h2, q_list, q, num_orb, nb, threshold):
    block = extract_q_block(h2, q_list, q, num_orb)   # (nb, nb, ng_raw)
    mat   = block.reshape(nb * nb, -1)                 # (nb^2, ng_raw)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    kept   = s > threshold
    n_kept = int(kept.sum())
    if n_kept == 0:
        compressed = np.zeros((nb, nb, 1), dtype=np.complex128)
        s_min = 0.0
        n_kept = 1
    else:
        compressed = (u[:, kept] @ np.diag(s[kept])).reshape(nb, nb, n_kept)
        s_min = float(s[kept][-1])
    return compressed, float(s[0]), s_min, n_kept


# ---------------------------------------------------------------------------
#  Main compression
# ---------------------------------------------------------------------------
def compress(h2_in, qlist_in, h2_out, num_orb, num_k, threshold, n_jobs):
    nb = num_orb * num_k

    # --- load H2 ---
    print(f"Loading H2 from  : {h2_in}")
    t0 = time()
    h2 = np.load(h2_in)
    print(f"  raw shape      : {h2.shape}  dtype={h2.dtype}  ({h2.nbytes/2**20:.0f} MB)")

    # Normalise axis order → (nb, nb, ng_raw)
    if h2.shape[1] == nb and h2.shape[2] == nb:
        h2 = np.moveaxis(h2, 0, -1)      # (ng_raw, nb, nb) → (nb, nb, ng_raw)
    elif h2.shape[0] == nb and h2.shape[1] == nb:
        pass                               # already (nb, nb, ng_raw)
    else:
        sys.exit(f"ERROR: cannot interpret H2 shape {h2.shape} with nb={nb}")

    h2 = h2.astype(np.complex128)
    ng_raw = h2.shape[2]
    print(f"  working shape  : {h2.shape}  (ng_raw={ng_raw})")

    # --- load Q_list ---
    print(f"\nLoading Q_list from: {qlist_in}")
    q_list = load_q_list(qlist_in, num_k)
    print(f"  shape          : {q_list.shape}  (should be {num_k**2} rows)")
    if len(q_list) != num_k ** 2:
        print(f"  WARNING: expected {num_k**2} rows, got {len(q_list)}")

    # --- parallel SVD over Q sectors ---
    print(f"\nSVD per momentum sector  (threshold={threshold}, n_jobs={n_jobs})")
    print(f"{'Q':>4}  {'n_kept':>8}  {'s_max':>10}  {'s_min_kept':>12}  {'MB':>6}")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_svd_one_q)(h2, q_list, q, num_orb, nb, threshold)
        for q in range(1, num_k + 1)
    )

    parts = []
    for q, (compressed, s_max, s_min, n_kept) in enumerate(results, start=1):
        parts.append(compressed)
        print(f"{q:>4}  {n_kept:>8}  {s_max:>10.4f}  {s_min:>12.2e}  {compressed.nbytes/2**20:>6.2f}")

    h2_zip = np.concatenate(parts, axis=2).astype(np.complex128)
    ng_total = h2_zip.shape[2]
    print(f"\nCompression summary")
    print(f"  Input  (nb, nb, ng_raw) : ({nb}, {nb}, {ng_raw})")
    print(f"  Output (nb, nb, ng_compressed) : {h2_zip.shape}")
    print(f"  Compression ratio       : {ng_raw / (ng_total / num_k):.1f}x per sector")
    print(f"  Memory raw → compressed : {h2.nbytes/2**20:.0f} MB → {h2_zip.nbytes/2**20:.1f} MB")

    np.save(h2_out, h2_zip)
    print(f"\nSaved: {h2_out}")
    print(f"\n{'='*50}")
    print(f"  Set in vafpy.in:  NGVEC : {ng_total}")
    print(f"{'='*50}")
    print(f"  (total runtime: {time()-t0:.1f} s)")
    return ng_total


def copy_h1(h1_in, h1_out):
    h1 = np.load(h1_in).astype(np.complex128)
    print(f"\nH1: {h1_in} {h1.shape} → {h1_out}")
    np.save(h1_out, h1)


if __name__ == "__main__":
    print("=" * 60)
    print("  compress_H2.py — H2 SVD compression for vafpy_v2")
    print("=" * 60)
    print(f"  num_orb={NUM_ORB}  num_k={NUM_K}  threshold={SVD_THRESH}  n_jobs={N_JOBS}")
    print()
    ng = compress(H2_IN, QLIST_IN, H2_OUT, NUM_ORB, NUM_K, SVD_THRESH, N_JOBS)
    copy_h1(H1_IN, H1_OUT)
    print()
    print("Next steps:")
    print(f"  1. Copy H2_zip.npy, H1_svd.npy, Q_list.npy to your run directory.")
    print(f"  2. Set in vafpy.in:")
    print(f"       NGVEC  : {ng}")
    print(f"       KPOINT : {NUM_K}")
    print(f"       NORB   : {NUM_ORB}")
    print(f"  3. mpirun -n <N> python3 run_kpts.py")
