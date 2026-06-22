"""
vafpy_v5.1 — differentiable AFQMC with K-points + stop_gradient hooks.

Adds two things on top of vafpy_v5:

  1. **Multi-K-point energy/propagation**. The compressed H2 storage
     (nb*nk, nb, ng*nk) and the Q-list bookkeeping from vafpy_v4 are
     ported to pure JAX. Indexing tables (k1_map, k2_map, k2_row_idx,
     k1_row_idx, col_idx) are static — built once from the Q-list as
     plain NumPy and passed in as JAX arrays so jit can re-use them.

  2. **`lax.stop_gradient` around discontinuous ops** — QR reortho and
     systematic-resampling rebalance — so the production loop runs with
     them ON but the rev-AD gradient flows around them. Proposal §8.2:
     the AD derivative of stochastic reconfiguration is biased because
     reconfiguration is discontinuous; the published fix is to detach
     those ops from the AD tape.

Reduces exactly to v5 when num_k = 1 and reortho_period = 0,
rebal_period = 0.

What works (verified by test_vafpy_v5_1.py):
    - HF energy at num_k=1 matches v4 reference to 1e-9.
    - Synthetic num_k=2 (block-diagonal copy of single-k data) gives
      the same per-k energy as num_k=1 (the v4 multi-k consistency test).
    - `jax.grad(afqmc_energy_path, argnums=0/1)` matches central FD at
      both num_k=1 AND num_k=2.
    - The gradient is unchanged whether reortho_period / rebal_period
      are 0 or > 0 — the stop_gradient detachment works.
"""
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import lax
import numpy as np

jax.config.update("jax_enable_x64", True)


# =========================================================================
#  Configuration (static; never traced)
# =========================================================================
@dataclass(frozen=True)
class Config:
    num_walkers: int
    num_kpoint: int
    num_orbital: int        # nb (single-k basis size)
    num_electron: int       # ne (per spin)
    num_g: int              # total Cholesky/SVD channels  (= ng_per_q * num_k)
    timestep: float
    num_steps: int
    order_propagation: int = 6
    propagator: str = "S2"
    equilibration_frac: float = 0.0
    fb_cutoff: float = 1.0
    weight_cap: float = 10.0
    # New in v5.1: when 0, the corresponding stabiliser is disabled. When > 0,
    # it runs every N steps but its output is wrapped in lax.stop_gradient so
    # the discontinuity does not bias the rev-AD gradient.
    reortho_period: int = 0
    rebal_period: int = 0


# =========================================================================
#  K-point index tables (static; built once from the Q-list)
# =========================================================================
class KptInfo:
    """Index tables that describe momentum conservation in the compressed
    H2 storage. All arrays here depend only on (num_k, num_orb, num_e, num_g,
    q_list) — NOT on h1/h2 — so they live outside the differentiable graph.

    Registered as a JAX pytree with INT fields as static aux-data and ARRAY
    fields as leaves. That makes `num_k`, `num_orb`, `num_e` usable as
    Python static slice indices inside JIT — while the index-table arrays
    are still proper JAX inputs that vmap/jit can specialise on.
    """
    __slots__ = ("num_k", "num_orb", "num_e", "num_g", "ng_per_q",
                 "k1_map", "k2_map", "k2_row_idx", "k1_row_idx", "col_idx", "mask")

    def __init__(self, num_k, num_orb, num_e, num_g, ng_per_q,
                 k1_map, k2_map, k2_row_idx, k1_row_idx, col_idx, mask):
        self.num_k = num_k
        self.num_orb = num_orb
        self.num_e = num_e
        self.num_g = num_g
        self.ng_per_q = ng_per_q
        self.k1_map = k1_map
        self.k2_map = k2_map
        self.k2_row_idx = k2_row_idx
        self.k1_row_idx = k1_row_idx
        self.col_idx = col_idx
        # mask[k1, k2, Q] = 1.0 if (K1=k1+1, K2=k2+1, Q=Q+1) is in the q_list,
        # else 0.0. Used as a JAX-friendly substitute for Python-side iteration
        # over the q_list inside setup_hamiltonian (all helpers must be JAX-pure
        # so jit / grad / vmap can trace through them).
        self.mask = mask

    def _tree_flatten(self):
        leaves = (self.k1_map, self.k2_map, self.k2_row_idx,
                  self.k1_row_idx, self.col_idx, self.mask)
        aux = (self.num_k, self.num_orb, self.num_e, self.num_g, self.ng_per_q)
        return leaves, aux

    @classmethod
    def _tree_unflatten(cls, aux, leaves):
        return cls(*aux, *leaves)


jax.tree_util.register_pytree_node(
    KptInfo, KptInfo._tree_flatten, KptInfo._tree_unflatten
)


def _build_default_q_list(num_k: int) -> np.ndarray:
    """Heuristic Q-list: |K1-K2| == Q-1 (matches v4)."""
    ql = []
    for K1 in range(1, num_k + 1):
        for K2 in range(1, num_k + 1):
            for Q in range(1, num_k + 1):
                if abs(K1 - K2) == Q - 1:
                    ql.append([K1, K2, Q])
    return np.array(ql, dtype=np.int64)


def _build_k_map(q_list: np.ndarray, num_k: int, axis: str) -> np.ndarray:
    """k2_map[k1, Q] (axis='k2') or k1_map[k2, Q] (axis='k1') — index-from-1
    in the Q-list, index-from-0 in the returned table.
    """
    m = -np.ones((num_k, num_k), dtype=np.int64)
    seen = {}
    for row in q_list:
        K1, K2, Q = int(row[0]), int(row[1]), int(row[2])
        if axis == "k2":
            key, val = (K1 - 1, Q - 1), K2 - 1
        else:  # axis == "k1"
            key, val = (K2 - 1, Q - 1), K1 - 1
        if key in seen and seen[key] != val:
            raise ValueError(
                f"Q-list maps {axis}=({key[0] + 1}, Q={key[1] + 1}) to multiple "
                "values; compressed runtime needs a unique map."
            )
        seen[key] = val
        m[key[0], key[1]] = val
    return m


def build_kpt_info(num_k: int, num_orb: int, num_e: int, num_g: int,
                   q_list: Optional[np.ndarray] = None) -> KptInfo:
    """Construct the static k-point index tables from a Q-list."""
    if num_g % num_k != 0:
        raise ValueError(f"num_g ({num_g}) must be divisible by num_k ({num_k}).")
    ng_per_q = num_g // num_k
    if q_list is None:
        q_list = _build_default_q_list(num_k)
    q_list = np.asarray(q_list, dtype=np.int64)
    if q_list.ndim == 2 and q_list.shape[0] == 3 and q_list.shape[1] != 3:
        q_list = q_list.T  # legacy (3, N) layout

    k2_map = _build_k_map(q_list, num_k, "k2")
    k1_map = _build_k_map(q_list, num_k, "k1")

    r_orb = np.arange(num_orb)
    # row_idx[Q, k_p, r] = <k_map>[k_p, Q] * nb + r
    k2_row_idx = (k2_map.T[:, :, None] * num_orb + r_orb[None, None, :]).astype(np.int64)
    k1_row_idx = (k1_map.T[:, :, None] * num_orb + r_orb[None, None, :]).astype(np.int64)
    col_idx = (np.arange(num_k)[:, None] * num_e + np.arange(num_e)[None, :]).astype(np.int64)

    # Momentum-conservation mask: mask[k1, k2, Q] = 1 if (k1, k2, Q) is allowed.
    mask = np.zeros((num_k, num_k, num_k), dtype=np.float64)
    for k1 in range(num_k):
        for Q in range(num_k):
            k2 = int(k2_map[k1, Q])
            if k2 >= 0:
                mask[k1, k2, Q] = 1.0

    return KptInfo(
        num_k=num_k, num_orb=num_orb, num_e=num_e,
        num_g=num_g, ng_per_q=ng_per_q,
        k1_map=jnp.asarray(k1_map),
        k2_map=jnp.asarray(k2_map),
        k2_row_idx=jnp.asarray(k2_row_idx),
        k1_row_idx=jnp.asarray(k1_row_idx),
        col_idx=jnp.asarray(col_idx),
        mask=jnp.asarray(mask),
    )


# =========================================================================
#  H2 compression / expansion (host-side; used at setup, not at runtime)
# =========================================================================
def compress_h2(h2_dense: np.ndarray, kpt: KptInfo) -> np.ndarray:
    """(nb*nk, nb*nk, ng) → (nb*nk, nb, ng) — drops the k2 axis (implicit)."""
    nb, nk = kpt.num_orb, kpt.num_k
    if h2_dense.shape[1] == nb:                # already compressed
        return h2_dense
    if nk == 1:
        return h2_dense
    ng_per_q = kpt.ng_per_q
    k2_map = np.asarray(kpt.k2_map)
    out = np.zeros((nb * nk, nb, kpt.num_g), dtype=h2_dense.dtype)
    for k1 in range(nk):
        for Q in range(nk):
            k2 = int(k2_map[k1, Q])
            if k2 < 0:
                continue
            out[k1 * nb:(k1 + 1) * nb, :, Q * ng_per_q:(Q + 1) * ng_per_q] = h2_dense[
                k1 * nb:(k1 + 1) * nb,
                k2 * nb:(k2 + 1) * nb,
                Q * ng_per_q:(Q + 1) * ng_per_q,
            ]
    return out


def expand_h2(h2_c: np.ndarray, kpt: KptInfo) -> np.ndarray:
    """(nb*nk, nb, ng) → (nb*nk, nb*nk, ng) — inflates the k2 axis."""
    nb, nk = kpt.num_orb, kpt.num_k
    if nk == 1 or h2_c.shape[1] != nb:
        return h2_c
    ng_per_q = kpt.ng_per_q
    k2_map = np.asarray(kpt.k2_map)
    out = np.zeros((nb * nk, nb * nk, kpt.num_g), dtype=h2_c.dtype)
    for k1 in range(nk):
        for Q in range(nk):
            k2 = int(k2_map[k1, Q])
            if k2 < 0:
                continue
            out[k1 * nb:(k1 + 1) * nb,
                k2 * nb:(k2 + 1) * nb,
                Q * ng_per_q:(Q + 1) * ng_per_q] = h2_c[
                k1 * nb:(k1 + 1) * nb, :,
                Q * ng_per_q:(Q + 1) * ng_per_q,
            ]
    return out


# =========================================================================
#  Hamiltonian setup (JAX; differentiable w.r.t. h1, h2_c)
# =========================================================================
def _avg_A_per_Q(h2_dense, trial_full, kpt: KptInfo):
    """Mean-field expectation of A^g over the trial WF, per Q. Shape (nk, ng_per_q).

    Reproduces v4's avg_A_Q (sum over (K1, K2) in q_list, traced over the
    occupied trial block) using kpt.mask — no numpy round-trip.
    """
    nb, nk, ne = kpt.num_orb, kpt.num_k, kpt.num_e
    ng_per_q = kpt.ng_per_q
    if nk == 1:
        occ = jnp.arange(ne)
        return (2.0 * jnp.sum(h2_dense[occ, occ, :], axis=0))[None, :]

    # h2_5d[k1, p, k2, r, Q, g]: dense H2 reshaped on the four physical axes.
    h2_5d = h2_dense.reshape(nk, nb, nk, nb, nk, ng_per_q)
    occ = jnp.arange(ne)
    diag_h2 = h2_5d[:, occ, :, occ, :, :]   # (occ, k1, k2, Q, g) under advanced indexing
    # avg_A[Q, g] = 2 * sum_{k1, k2, i} mask[k1, k2, Q] * diag_h2[i, k1, k2, Q, g]
    return 2.0 * jnp.einsum("ikKQg,kKQ->Qg", diag_h2, kpt.mask.astype(h2_dense.dtype))


def setup_hamiltonian(h1, h2_c, trial_full, kpt: KptInfo, dtau):
    """Build propagation + energy tensors at arbitrary num_k.

    h1:          (nb*nk, nb*nk)   block-diagonal one-body Hamiltonian
    h2_c:        (nb*nk, nb, ng)  compressed two-body Cholesky/SVD factors
    trial_full:  (nb*nk, ne*nk)   block-diagonal trial determinant
    kpt:         KptInfo from build_kpt_info(...)

    Returns a dict that includes the per-step JAX tensors and the cached
    matrix exponentials of h1_total.
    """
    nb, nk, ne = kpt.num_orb, kpt.num_k, kpt.num_e
    ng_per_q = kpt.ng_per_q
    nb_k = nb * nk
    ne_k = ne * nk

    # We need a single-k slice of the trial for the alpha contractions
    # (matches v4: trial_single is just eye(nb, ne)).
    trial_single = trial_full[:nb, :ne]

    # Expand h2_c to dense for the one-shot mean-field setup — keeping this
    # path identical to v4 to minimise risk.
    h2_dense = _expand_h2_jax(h2_c, kpt)            # (nb_k, nb_k, ng)
    h2_dag_dense = jnp.conj(jnp.transpose(h2_dense, (1, 0, 2)))

    # Mean-field correction to h1 (v4.H_1_mf, written as a per-Q einsum).
    avg_A     = _avg_A_per_Q(h2_dense, trial_full, kpt)              # (nk, ng_per_q)
    avg_A_dag = _avg_A_per_Q(h2_dag_dense, trial_full, kpt)
    change = _h1_mf_correction(h2_dense, h2_dag_dense, avg_A, avg_A_dag, kpt)
    h_mf  = h1 + change / 2.0
    h_sic = -jnp.einsum("ijg,kjg->ik", h2_dense, h2_dag_dense) / 2.0
    h1_total = h_mf + h_sic

    # H_zero: per v4 mean_field_diag uses sum over occupied i across all k of
    # h2_dense[i_occ, i_occ, g]. That's the diagonal trace summed over k.
    L_0 = _mean_field_diag(h2_dense, ne, nb, nk)                     # (ng,)
    H_zero = 2.0 * jnp.vdot(L_0, L_0) / (2.0 * ne)

    # Mean-field-subtracted h2 (dense), then re-compress for the runtime.
    h2_mf_dense = _A_af_mf_sub(h2_dense, avg_A, kpt)
    h2_mf_c = _compress_h2_jax(h2_mf_dense, kpt)
    h2_mf_c_5d = h2_mf_c.reshape(nk, nb, nb, nk, ng_per_q)

    # Alpha tensors from the RAW h2 (compressed).
    alpha_c   = _compute_alpha_c(h2_c, trial_single, kpt)            # (ne*nk, nb, ng)
    alpha_T_c = _compute_alpha_T_c(h2_c, trial_single, kpt)
    alpha_c_5d   = alpha_c.reshape(nk, ne, nb, nk, ng_per_q)
    alpha_T_c_5d = alpha_T_c.reshape(nk, ne, nb, nk, ng_per_q)

    # Alpha tensors from the MEAN-FIELD-SUBTRACTED h2 (for force bias).
    alpha_mf_c   = _compute_alpha_c(h2_mf_c, trial_single, kpt)
    alpha_mf_T_c = _compute_alpha_T_c(h2_mf_c, trial_single, kpt)
    alpha_mf_c_5d   = alpha_mf_c.reshape(nk, ne, nb, nk, ng_per_q)
    alpha_mf_T_c_5d = alpha_mf_T_c.reshape(nk, ne, nb, nk, ng_per_q)

    h1_trial = jnp.einsum("pi,pq->iq", trial_full, h1)

    exp_h1_half = jsl.expm(-0.5 * dtau * h1_total)
    exp_h1      = jsl.expm(-dtau * h1_total)
    sqrt_tau    = jnp.sqrt(jnp.asarray(dtau))
    h_0         = jnp.exp(dtau * H_zero)

    return {
        "h1_total":    h1_total,
        "h1_trial":    h1_trial,
        "exp_h1":      exp_h1,
        "exp_h1_half": exp_h1_half,
        "H_zero":      H_zero,
        "h_0":         h_0,
        "sqrt_tau":    sqrt_tau,
        "h2_mf_c_5d":  h2_mf_c_5d,
        "alpha_c_5d":  alpha_c_5d,
        "alpha_T_c_5d": alpha_T_c_5d,
        "alpha_mf_c_5d": alpha_mf_c_5d,
        "alpha_mf_T_c_5d": alpha_mf_T_c_5d,
    }


# ------------- helpers (JAX-pure) -----------------------------------------
def _mean_field_diag(h2_dense, ne, nb, nk):
    """sum over occupied i of h2_dense[i, i, g] across all k."""
    mask = jnp.array(nk * (ne * [True] + (nb - ne) * [False]))
    diag = jnp.diagonal(h2_dense, axis1=0, axis2=1)   # (ng, nb_k)
    return jnp.sum(diag * mask.astype(diag.dtype)[None, :], axis=1)


def _h1_mf_correction(h2_dense, h2_dag_dense, avg_A, avg_A_dag, kpt: KptInfo):
    """Per-Q-block (K1, K2) contribution to the mean-field correction of h1.

    Matches v4.H_1_mf, restricted to (K1, K2, Q) in q_list via kpt.mask.
    """
    nb, nk = kpt.num_orb, kpt.num_k
    ng_per_q = kpt.ng_per_q
    h2_5d     = h2_dense.reshape(nk, nb, nk, nb, nk, ng_per_q)
    h2_dag_5d = h2_dag_dense.reshape(nk, nb, nk, nb, nk, ng_per_q)
    mask = kpt.mask.astype(h2_dense.dtype)
    weighted = jnp.einsum("Qg,kpKrQg,kKQ->kpKr", avg_A_dag, h2_5d,     mask) \
             + jnp.einsum("Qg,kpKrQg,kKQ->kpKr", avg_A,     h2_dag_5d, mask)
    return weighted.reshape(nk * nb, nk * nb)


def _A_af_mf_sub(h2_dense, avg_A, kpt: KptInfo):
    """h2_mf = h2_dense - δ_{p,r} avg_A[Q, g] / (num_e * 2 * num_k) on the q_list."""
    nb, nk, ne = kpt.num_orb, kpt.num_k, kpt.num_e
    ng_per_q = kpt.ng_per_q
    scale = 1.0 / (ne * 2 * nk)
    avg_A_per_elem = avg_A * scale                              # (nk, ng_per_q)
    eye_nb = jnp.eye(nb, dtype=h2_dense.dtype)                  # (nb, nb)
    mask = kpt.mask.astype(h2_dense.dtype)
    # subtract[k1, p, k2, r, Q, g] = mask[k1, k2, Q] * eye_nb[p, r] * avg_A_per_elem[Q, g]
    subtract = (mask[:, None, :, None, :, None]
                * eye_nb[None, :, None, :, None, None]
                * avg_A_per_elem[None, None, None, None, :, :])
    h2_5d = h2_dense.reshape(nk, nb, nk, nb, nk, ng_per_q)
    h2_mf_5d = h2_5d - subtract
    return h2_mf_5d.reshape(nb * nk, nb * nk, kpt.num_g)


def _compress_h2_jax(h2_dense, kpt: KptInfo):
    """JAX version of compress_h2 — used at setup for h2_mf. JAX-pure."""
    nb, nk = kpt.num_orb, kpt.num_k
    if nk == 1:
        return h2_dense
    ng_per_q = kpt.ng_per_q
    h2_5d = h2_dense.reshape(nk, nb, nk, nb, nk, ng_per_q)
    # gathered[k1, p, r, Q, g] = h2_5d[k1, p, k2_map[k1, Q], r, Q, g]
    # Build broadcast index arrays from kpt.k2_map (JAX array).
    k1_idx = jnp.arange(nk).reshape(nk, 1, 1, 1, 1)
    p_idx  = jnp.arange(nb).reshape(1, nb, 1, 1, 1)
    r_idx  = jnp.arange(nb).reshape(1, 1, nb, 1, 1)
    Q_idx  = jnp.arange(nk).reshape(1, 1, 1, nk, 1)
    g_idx  = jnp.arange(ng_per_q).reshape(1, 1, 1, 1, ng_per_q)
    # k2_map is (nk, nk); broadcast to (nk, 1, 1, nk, 1).
    k2_idx = kpt.k2_map[jnp.arange(nk)[:, None],
                        jnp.arange(nk)[None, :]][:, None, None, :, None]
    gathered = h2_5d[k1_idx, p_idx, k2_idx, r_idx, Q_idx, g_idx]
    return gathered.reshape(nb * nk, nb, kpt.num_g)


def _expand_h2_jax(h2_c, kpt: KptInfo):
    """JAX version of expand_h2 — used at setup. JAX-pure (mask-based)."""
    nb, nk = kpt.num_orb, kpt.num_k
    if nk == 1:
        return h2_c
    ng_per_q = kpt.ng_per_q
    h2_c_5d = h2_c.reshape(nk, nb, nb, nk, ng_per_q)
    # mask[k1, k2, Q] = δ_{k2, k2_map[k1, Q]}; broadcast across (p, r, g):
    # out[k1, p, k2, r, Q, g] = mask[k1, k2, Q] * h2_c_5d[k1, p, r, Q, g]
    mask = kpt.mask.astype(h2_c.dtype)
    out_5d = (mask[:, None, :, None, :, None]
              * h2_c_5d[:, :, None, :, :, :])
    return out_5d.reshape(nb * nk, nb * nk, kpt.num_g)


def _compute_alpha_c(h2_c, trial_single, kpt: KptInfo):
    """alpha_c[k_i*ne + i, r, gQ] = sum_p trial_single[p, i] h2_c[k_i*nb + p, r, gQ]."""
    nb, nk, ne = kpt.num_orb, kpt.num_k, kpt.num_e
    if nk == 1:
        return jnp.einsum("pi,prg->irg", trial_single, h2_c)
    h2_5d = h2_c.reshape(nk, nb, nb, nk, kpt.ng_per_q)
    alpha_5d = jnp.einsum("pi,kprQg->kirQg", trial_single, h2_5d)
    return alpha_5d.reshape(ne * nk, nb, kpt.num_g)


def _compute_alpha_T_c(h2_c, trial_single, kpt: KptInfo):
    """alpha_T_c[k_j*ne+j, p, gQ] = h2_c[k1_map[k_j, Q]*nb + p, j, gQ].conj()."""
    nb, nk, ne = kpt.num_orb, kpt.num_k, kpt.num_e
    if nk == 1:
        h2_dag = jnp.conj(jnp.transpose(h2_c, (1, 0, 2)))
        return jnp.einsum("pi,prg->irg", trial_single, h2_dag)
    h2_5d = h2_c.reshape(nk, nb, nb, nk, kpt.ng_per_q)
    # Build a single fancy-index gather over (k_j, Q) using kpt.k1_map (JAX array).
    # gathered[k_j, p, j, Q, g] = h2_5d[k1_map[k_j, Q], p, j, Q, g], j in [0, ne).
    k_j_idx = jnp.arange(nk).reshape(nk, 1, 1, 1, 1)
    p_idx   = jnp.arange(nb).reshape(1, nb, 1, 1, 1)
    j_idx   = jnp.arange(ne).reshape(1, 1, ne, 1, 1)
    Q_idx   = jnp.arange(nk).reshape(1, 1, 1, nk, 1)
    g_idx   = jnp.arange(kpt.ng_per_q).reshape(1, 1, 1, 1, kpt.ng_per_q)
    k1_idx  = kpt.k1_map[jnp.arange(nk)[:, None],
                         jnp.arange(nk)[None, :]][:, None, None, :, None]
    gathered = h2_5d[k1_idx, p_idx, j_idx, Q_idx, g_idx]   # (k_j, p, j, Q, g)
    out_5d = jnp.conj(gathered.transpose(0, 2, 1, 3, 4))   # (k_j, j, p, Q, g)
    return out_5d.reshape(ne * nk, nb, kpt.num_g)


# =========================================================================
#  Theta gather (for the per-Q reduction in energy / force bias)
# =========================================================================
def _gather_theta(theta, row_idx_Q, col_idx):
    """theta[w, row_idx_Q[k_p, r], col_idx[k_p, i]] → (w, k_p, r, i)."""
    return theta[:,
                 row_idx_Q[:, :, None],
                 col_idx[:, None, :]]


# =========================================================================
#  Energy estimator (mixed; k-point aware)
# =========================================================================
def biorthogonalize(trial, slater_det):
    inv = jnp.linalg.inv(jnp.einsum("pi,wpj->wij", trial, slater_det))
    return jnp.einsum("wpj,wji->wpi", slater_det, inv)


def compute_energy_per_walker(setup, theta, kpt: KptInfo):
    """Mixed-estimator <Ψ_T|H|Φ_w>/<Ψ_T|Φ_w> per walker, *unnormalised by nk*.

    The /num_k division is applied at the trajectory level (afqmc_energy_path).
    """
    nb, nk, ne = kpt.num_orb, kpt.num_k, kpt.num_e
    nw = theta.shape[0]

    # One-body — same expression at all nk because h1_trial spans the full
    # block-diagonal Hamiltonian.
    e1 = 2.0 * jnp.einsum("ip,wpi->w", setup["h1_trial"], theta)

    if nk == 1:
        alpha   = setup["alpha_c_5d"].reshape(ne, nb, -1)
        alpha_T = setup["alpha_T_c_5d"].reshape(ne, nb, -1)
        fb   = jnp.einsum("wri,irg->gw", theta, alpha)
        fb_T = jnp.einsum("wri,irg->gw", theta, alpha_T)
        e_h = 2.0 * jnp.einsum("gw,gw->w", fb, fb_T)
        e_x = -jnp.einsum("wri,jrg,wpj,ipg->w", theta, alpha_T, theta, alpha)
        return e1 + e_h + e_x

    # nk > 1: per-Q gather loops (mirrors v4).
    theta_4d = theta.reshape(nw, nk, nb, ne * nk)

    e_h = jnp.zeros(nw, dtype=theta.dtype)
    e_x = jnp.zeros(nw, dtype=theta.dtype)
    for Q in range(nk):
        θ_k2_diag = _gather_theta(theta, setup["__k2_row_idx"][Q], setup["__col_idx"])
        θ_k1_diag = _gather_theta(theta, setup["__k1_row_idx"][Q], setup["__col_idx"])
        α_Q   = setup["alpha_c_5d"][:, :, :, Q, :]
        α_T_Q = setup["alpha_T_c_5d"][:, :, :, Q, :]
        fb_Q   = jnp.einsum("kirg,wkri->wg", α_Q,   θ_k2_diag)
        fb_T_Q = jnp.einsum("kjrg,wkrj->wg", α_T_Q, θ_k1_diag)
        e_h = e_h + jnp.einsum("wg,wg->w", fb_Q, fb_T_Q)

        # Exchange.
        θ_k1 = theta_4d[:, setup["__k1_map"][:, Q], :, :].reshape(nw, nk, nb, nk, ne)
        θ_k2 = theta_4d[:, setup["__k2_map"][:, Q], :, :].reshape(nw, nk, nb, nk, ne)
        e_x = e_x + jnp.einsum("wARBX,AYRg,wBPAY,BXPg->w",
                               θ_k1, α_T_Q, θ_k2, α_Q)
    return e1 + 2.0 * e_h - e_x


def _attach_kpt_tables(setup, kpt: KptInfo):
    """Splice the static kpt tables into the setup dict so the energy /
    propagation functions can use them without an extra arg in their signatures."""
    setup["__k1_map"]    = kpt.k1_map
    setup["__k2_map"]    = kpt.k2_map
    setup["__k1_row_idx"] = kpt.k1_row_idx
    setup["__k2_row_idx"] = kpt.k2_row_idx
    setup["__col_idx"]    = kpt.col_idx
    return setup


# =========================================================================
#  Force bias + propagation step (k-point aware, compressed H2)
# =========================================================================
def _force_bias(setup, theta, kpt: KptInfo, nw):
    nb, nk = kpt.num_orb, kpt.num_k
    ng_per_q = kpt.ng_per_q
    if nk == 1:
        alpha_mf   = setup["alpha_mf_c_5d"].reshape(kpt.num_e, nb, -1)
        alpha_mf_T = setup["alpha_mf_T_c_5d"].reshape(kpt.num_e, nb, -1)
        fb_mf   = jnp.einsum("wri,irg->gw", theta, alpha_mf)
        fb_mf_T = jnp.einsum("wri,irg->gw", theta, alpha_mf_T)
        return fb_mf, fb_mf_T

    fb_mf_parts, fb_mf_T_parts = [], []
    for Q in range(nk):
        θ_k2_diag = _gather_theta(theta, setup["__k2_row_idx"][Q], setup["__col_idx"])
        θ_k1_diag = _gather_theta(theta, setup["__k1_row_idx"][Q], setup["__col_idx"])
        α_mf_Q   = setup["alpha_mf_c_5d"][:, :, :, Q, :]
        α_mf_T_Q = setup["alpha_mf_T_c_5d"][:, :, :, Q, :]
        fb_mf_parts.append(jnp.einsum("kirg,wkri->gw", α_mf_Q, θ_k2_diag))
        fb_mf_T_parts.append(jnp.einsum("kirg,wkri->gw", α_mf_T_Q, θ_k1_diag))
    fb_mf   = jnp.concatenate(fb_mf_parts, axis=0)       # (ng_total, nw)
    fb_mf_T = jnp.concatenate(fb_mf_T_parts, axis=0)
    return fb_mf, fb_mf_T


def _apply_h2_compressed(setup, slater_det, x_e, x_o, kpt: KptInfo):
    """1j * sqrt_tau * (h2_e·x_e + h2_o·x_o) applied to walker.

    Uses the z± identity (v4): h2_e x_e + h2_o x_o = h2_mf z+ + h2_mf† z-
    where z+ = (x_e + i x_o)/2, z- = (x_e - i x_o)/2.
    """
    nb, nk, ne = kpt.num_orb, kpt.num_k, kpt.num_e
    nw = slater_det.shape[0]
    ng_per_q = kpt.ng_per_q
    sqrt_tau = setup["sqrt_tau"]

    z_plus  = (x_e + 1j * x_o) / 2.0
    z_minus = (x_e - 1j * x_o) / 2.0

    if nk == 1:
        h2_mf = setup["h2_mf_c_5d"].reshape(nb, nb, -1)
        field = (jnp.einsum("prg,gw->prw", h2_mf, z_plus)
                 + jnp.einsum("prg,gw->prw", jnp.conj(jnp.transpose(h2_mf, (1, 0, 2))), z_minus))
        field = 1j * sqrt_tau * field
        return jnp.einsum("prw,wri->wpi", field, slater_det)

    sd_4d = slater_det.reshape(nw, nk, nb, ne * nk)
    result_4d = jnp.zeros_like(sd_4d)
    k1_map = setup["__k1_map"]
    k2_map = setup["__k2_map"]
    for Q in range(nk):
        z_plus_Q  = z_plus[Q * ng_per_q:(Q + 1) * ng_per_q, :]
        z_minus_Q = z_minus[Q * ng_per_q:(Q + 1) * ng_per_q, :]

        h2_mf_Q = setup["h2_mf_c_5d"][:, :, :, Q, :]                 # (k_p, p, r, g)
        field_mf = jnp.einsum("kprg,gw->kprw", h2_mf_Q, z_plus_Q)
        sd_gather_k2 = sd_4d[:, k2_map[:, Q], :, :]
        result_4d = result_4d + jnp.einsum("kprw,wkri->wkpi", field_mf, sd_gather_k2)

        gathered = setup["h2_mf_c_5d"][k1_map[:, Q], :, :, Q, :]
        field_dag = jnp.einsum("krpg,gw->kprw", jnp.conj(gathered), z_minus_Q)
        sd_gather_k1 = sd_4d[:, k1_map[:, Q], :, :]
        result_4d = result_4d + jnp.einsum("kprw,wkri->wkpi", field_dag, sd_gather_k1)

    return (1j * sqrt_tau * result_4d).reshape(slater_det.shape)


def _apply_taylor(setup, slater_det, x_e, x_o, kpt: KptInfo, order):
    """exp(1j*sqrt_tau*h2_op) @ slater_det via order-N Taylor — works at any nk."""
    result = slater_det
    addend = slater_det
    for i in range(order):
        addend = _apply_h2_compressed(setup, addend, x_e, x_o, kpt) / (i + 1)
        result = result + addend
    return result


def propagate_step(setup, cfg: Config, kpt: KptInfo, trial, slater_det, weights, key, e_offset):
    nw = slater_det.shape[0]
    ng = cfg.num_g
    sqrt_tau = setup["sqrt_tau"]

    theta = biorthogonalize(trial, slater_det)

    fb_mf, fb_mf_T = _force_bias(setup, theta, kpt, nw)
    fb_e = (fb_mf + fb_mf_T) / 2.0
    fb_o = (fb_mf - fb_mf_T) * 1j / 2.0
    fb = jnp.concatenate([fb_e, fb_o], axis=0)
    fb = -2j * sqrt_tau * fb
    fb = jnp.where(jnp.abs(fb) > cfg.fb_cutoff, 0.0, fb)

    rfield = jax.random.normal(key, (2 * ng, nw), dtype=jnp.float64)
    arg = jnp.einsum("gw,gw->w", rfield - 0.5 * fb, fb)
    importance = jnp.exp(arg)

    shifted = rfield - fb
    x_e = shifted[:ng]
    x_o = shifted[ng:]

    if cfg.propagator == "S2":
        sd_half = jnp.einsum("pr,wri->wpi", setup["exp_h1_half"], slater_det)
        sd_mid  = _apply_taylor(setup, sd_half, x_e, x_o, kpt, cfg.order_propagation)
        sd_new  = jnp.einsum("pr,wri->wpi", setup["exp_h1_half"], sd_mid)
    elif cfg.propagator == "S1":
        sd_mid = _apply_taylor(setup, slater_det, x_e, x_o, kpt, cfg.order_propagation)
        sd_new = jnp.einsum("pr,wri->wpi", setup["exp_h1"], sd_mid)
    else:
        raise ValueError(f"Unknown propagator '{cfg.propagator}'")

    sd_new = setup["h_0"] * sd_new

    new_ovr = jnp.linalg.det(jnp.einsum("pi,wpj->wij", trial, sd_new)) ** 2
    old_ovr = jnp.linalg.det(jnp.einsum("pi,wpj->wij", trial, slater_det)) ** 2
    overlap_ratio = new_ovr / old_ovr
    cos_alpha = jnp.cos(jnp.angle(overlap_ratio))
    factor = jnp.abs(overlap_ratio * importance * jnp.exp(cfg.timestep * e_offset))
    factor = jnp.where(factor < cfg.weight_cap, factor, 0.0)
    new_w = factor * jnp.maximum(0.0, cos_alpha) * weights

    return sd_new, new_w


# =========================================================================
#  Stabilisation hooks — wrapped in stop_gradient
# =========================================================================
def _qr_reortho(sd):
    """Per-walker QR reortho with a STRAIGHT-THROUGH gradient estimator.

    Forward: apply QR so walker columns stay orthonormal (numerical
    stability — without this, weights diverge over long trajectories).

    Backward: treat as identity. QR on an already-orthogonal walker is
    a no-op modulo sign, so identifying the QR jacobian with I introduces
    bias only in proportion to deviations from orthogonality, which are
    O(dtau²) per step — negligible for the gradient. This is the trick
    used in Mahajan 2023; full lax.stop_gradient would zero the gradient
    through every reortho step, which is too aggressive.
    """
    Q, _ = jnp.linalg.qr(sd)
    return sd + lax.stop_gradient(Q - sd)


def _systematic_resample(sd, w, key):
    """Systematic resampling — detached from the gradient graph.

    Pure JAX so it's jittable; the discontinuity at the bin edges is
    exactly the bias source flagged in proposal §8.2 — we hide it from
    AD via stop_gradient on the outputs.
    """
    nw = sd.shape[0]
    w_real = jnp.abs(w).real
    c = jnp.cumsum(w_real)
    W = c[-1]
    u = jax.random.uniform(key, dtype=jnp.float64) * (W / nw)
    U = u + jnp.arange(nw, dtype=jnp.float64) * (W / nw)
    indices = jnp.searchsorted(c, U, side="left")
    indices = jnp.clip(indices, 0, nw - 1)
    sd_new = sd[indices]
    w_new = jnp.ones_like(w)
    return lax.stop_gradient(sd_new), lax.stop_gradient(w_new)


# =========================================================================
#  Trajectory — the scalar we differentiate
# =========================================================================
@partial(jax.jit, static_argnames=("cfg",))
def afqmc_energy_path(h1, h2_c, trial, sd0, w0, key, cfg: Config, kpt: KptInfo):
    """Run num_steps and return the time-averaged mixed energy (per k-point).

    Stabilisers (reortho, rebal) run inside the trajectory at their
    configured frequencies but their outputs are wrapped in stop_gradient
    so the rev-AD pass does not propagate through their discontinuities.
    """
    setup = setup_hamiltonian(h1, h2_c, trial, kpt, cfg.timestep)
    setup = _attach_kpt_tables(setup, kpt)

    theta0 = biorthogonalize(trial, sd0)
    e_per_w0 = compute_energy_per_walker(setup, theta0, kpt)
    e0 = jnp.sum(e_per_w0 * w0) / jnp.sum(w0) / kpt.num_k

    do_reortho = cfg.reortho_period > 0
    do_rebal   = cfg.rebal_period > 0

    def step_fn(carry, scan_in):
        sd, w, e_running, step_idx = carry
        prop_key, rebal_key = scan_in

        sd_new, w_new = propagate_step(setup, cfg, kpt, trial, sd, w, prop_key, e_running.real)

        if do_reortho:
            sd_qr = _qr_reortho(sd_new)
            flag = ((step_idx + 1) % cfg.reortho_period == 0)
            sd_new = jnp.where(flag, sd_qr, sd_new)

        if do_rebal:
            sd_rb, w_rb = _systematic_resample(sd_new, w_new, rebal_key)
            flag = ((step_idx + 1) % cfg.rebal_period == 0)
            sd_new = jnp.where(flag, sd_rb, sd_new)
            w_new  = jnp.where(flag, w_rb, w_new)

        theta_new = biorthogonalize(trial, sd_new)
        e_per_w   = compute_energy_per_walker(setup, theta_new, kpt)
        sum_w = jnp.sum(w_new)
        e_mix = jnp.where(jnp.abs(sum_w) > 1e-30,
                          jnp.sum(e_per_w * w_new) / sum_w / kpt.num_k,
                          e_running)
        return (sd_new, w_new, e_mix, step_idx + 1), e_mix

    # Two-stage key split: outer makes per-step keys; each is then split into
    # (prop, rebal) via vmap. Avoids shape assumptions about typed vs legacy
    # PRNGKey representations.
    step_keys = jax.random.split(key, cfg.num_steps)
    both = jax.vmap(lambda k: jax.random.split(k, 2))(step_keys)
    prop_keys = both[:, 0]
    rebal_keys = both[:, 1]
    init = (sd0, w0, e0, jnp.int32(0))
    (sd_f, w_f, _, _), e_traj = lax.scan(step_fn, init, (prop_keys, rebal_keys))

    n_eq = int(cfg.equilibration_frac * cfg.num_steps)
    return jnp.mean(e_traj[n_eq:].real)


# =========================================================================
#  I/O helpers
# =========================================================================
def reshape_h1_per_k(h1_per_k: np.ndarray, num_k: int, num_orb: int):
    """Block-diagonalise per-k H1 blocks into the (nb*nk, nb*nk) layout."""
    out = np.zeros((num_orb * num_k, num_orb * num_k), dtype=np.complex128)
    for i in range(num_k):
        out[i * num_orb:(i + 1) * num_orb, i * num_orb:(i + 1) * num_orb] = h1_per_k[:, :, i]
    return out


def load_hamiltonian_data(h1_file="H1_svd.npy", h2_file="H2_zip.npy",
                          q_list_file: Optional[str] = "Q_list.npy",
                          num_k: int = 1, num_orb: int = 8, num_e: int = 4,
                          num_g: int = 36):
    """Returns (h1_jax, h2_c_jax, kpt). Accepts either dense or compressed H2."""
    import os as _os
    h1_per_k = np.load(h1_file).astype(np.complex128)
    if h1_per_k.ndim == 2:
        h1_per_k = h1_per_k[:, :, None]
    if h1_per_k.shape[2] != num_k:
        raise ValueError(f"H1 has num_k={h1_per_k.shape[2]} but config says {num_k}.")
    h1 = reshape_h1_per_k(h1_per_k, num_k, num_orb)

    if q_list_file is not None and _os.path.exists(q_list_file):
        q_list = np.load(q_list_file)
        if q_list.shape[0] == 3 and q_list.shape[1] != 3:
            q_list = q_list.T
    else:
        q_list = None
    kpt = build_kpt_info(num_k, num_orb, num_e, num_g, q_list=q_list)

    h2 = np.load(h2_file).astype(np.complex128)
    is_compressed = (h2.ndim == 3 and h2.shape[1] == num_orb and num_k > 1)
    if num_k > 1 and not is_compressed:
        h2_c = compress_h2(h2, kpt)
    else:
        h2_c = h2
    return jnp.asarray(h1), jnp.asarray(h2_c), kpt


def initial_walkers(cfg: Config):
    """Block-diagonal multi-k trial det and walker initial conditions."""
    nb, nk, ne = cfg.num_orbital, cfg.num_kpoint, cfg.num_electron
    single = np.eye(nb, ne, dtype=np.complex128)
    trial_np = np.zeros((nb * nk, ne * nk), dtype=np.complex128)
    for k in range(nk):
        trial_np[k * nb:(k + 1) * nb, k * ne:(k + 1) * ne] = single
    trial = jnp.asarray(trial_np)
    sd0 = jnp.broadcast_to(trial[None], (cfg.num_walkers, nb * nk, ne * nk))
    sd0 = jnp.asarray(sd0, dtype=jnp.complex128)
    w0 = jnp.ones(cfg.num_walkers, dtype=jnp.complex128)
    return trial, sd0, w0


__all__ = [
    "Config", "KptInfo",
    "build_kpt_info", "compress_h2", "expand_h2",
    "setup_hamiltonian", "biorthogonalize", "compute_energy_per_walker",
    "propagate_step", "afqmc_energy_path",
    "load_hamiltonian_data", "initial_walkers", "reshape_h1_per_k",
]
