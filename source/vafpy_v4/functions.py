"""
vafpy_v2: unified GPU/CPU AFQMC with k-point support.

Backend abstraction (NumPy / JAX / CuPy) follows the vafpy_v1 design.
K-point physics (mean-field subtraction, Q-list, block-diagonal trial)
follows the legacy reference in Code/opt.
"""
from dataclasses import dataclass
import os
import types
from time import time

import numpy as np
import scipy
from scipy.linalg import expm, block_diag
from mpi4py import MPI
from opt_einsum import contract, contract_expression


# ---------- optional accelerators (only imported if requested) -------------
def _try_import_jax():
    import jax
    import jax.numpy as jnp
    return jax, jnp


def _try_import_cupy():
    import cupy as cp
    return cp


# =========================================================================
#  K-point helpers (kept close to the reference implementation in opt/)
# =========================================================================
def reshape_H1(H1, num_k, num_orb):
    """Build block-diagonal H1 from per-k blocks.

    Input H1 has shape (num_orb, num_orb, num_k); output is
    (num_orb*num_k, num_orb*num_k) with each k-block on the diagonal.
    """
    h1 = np.zeros([num_orb * num_k, num_orb * num_k], dtype=np.complex128)
    for i in range(num_k):
        h1[i * num_orb:(i + 1) * num_orb, i * num_orb:(i + 1) * num_orb] = H1[:, :, i]
    return h1


def get_q_list(q_list, q_selected):
    return q_list[q_list[:, 2] == q_selected]


def get_k1s_k2s(q_list, q_selected):
    sub = get_q_list(q_list, q_selected)
    return list(zip(sub[:, 0], sub[:, 1]))


def get_A_k1_k2(h2, k1_idx, k2_idx, num_orb):
    return h2[(k1_idx - 1) * num_orb:k1_idx * num_orb,
              (k2_idx - 1) * num_orb:k2_idx * num_orb, :]


def get_alpha_k1_k2(trial_0, h2, k1_idx, k2_idx, num_orb):
    A_Q = get_A_k1_k2(h2, k1_idx, k2_idx, num_orb)
    return np.einsum("ip,prG->irG", trial_0.T, A_Q)


def overlap(left, right):
    return np.dot(left.T, right)


def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial, walker)))


def avg_A_Q(trial_0, trial, h2, q_list, q_selected, num_orb, num_e):
    """Average of H2 over the trial WF for a specific Q (vector over G)."""
    K1s_K2s = get_k1s_k2s(q_list, q_selected)
    theta_full = theta(trial, trial)
    result = np.zeros(h2.shape[2], dtype=np.complex128)
    for K1, K2 in K1s_K2s:
        alpha = get_alpha_k1_k2(trial_0, h2, K1, K2, num_orb)
        block = theta_full[(K2 - 1) * num_orb:K2 * num_orb,
                           (K1 - 1) * num_e:K1 * num_e]
        result += contract("iiG->G",
                           contract("nrG,rm->nmG", alpha, block))
    return 2 * result


def A_af_MF_sub(trial_0, trial, h2, q_list, num_k, num_orb, num_e):
    """Mean-field subtracted two-body Hamiltonian."""
    avg_A_mat = np.zeros_like(h2)
    for Q in range(1, num_k + 1):
        avg_A_vec_Q = avg_A_Q(trial_0, trial, h2, q_list, Q, num_orb, num_e)
        K1s_K2s = get_k1s_k2s(q_list, Q)
        for K1, K2 in K1s_K2s:
            for g in range(h2.shape[2]):
                for r in range(num_orb):
                    avg_A_mat[(K1 - 1) * num_orb + r][(K2 - 1) * num_orb + r][g] = avg_A_vec_Q[g]
    return h2 - avg_A_mat / num_e / 2 / num_k


def H_1_mf(trial_0, trial, h2, h2_dagger, q_list, h1, num_k, num_orb, num_e):
    """Mean-field correction to one-body Hamiltonian."""
    change = np.zeros_like(h1, dtype=np.complex128)
    for Q in range(1, num_k + 1):
        avg_A_vec_Q = avg_A_Q(trial_0, trial, h2, q_list, Q, num_orb, num_e)
        avg_A_vec_Q_dag = avg_A_Q(trial_0, trial, h2_dagger, q_list, Q, num_orb, num_e)
        K1s_K2s = get_k1s_k2s(q_list, Q)
        for K1, K2 in K1s_K2s:
            block_h2 = h2[(K1 - 1) * num_orb:K1 * num_orb,
                          (K2 - 1) * num_orb:K2 * num_orb, :]
            block_h2_dag = h2_dagger[(K1 - 1) * num_orb:K1 * num_orb,
                                     (K2 - 1) * num_orb:K2 * num_orb, :]
            change[(K1 - 1) * num_orb:K1 * num_orb,
                   (K2 - 1) * num_orb:K2 * num_orb] = (
                contract("rpG->rp",
                         contract("G,rpG->rpG", avg_A_vec_Q_dag, block_h2)
                         + contract("G,rpG->rpG", avg_A_vec_Q, block_h2_dag))
            )
    return h1 + change / 2


def mean_field_diag(h2, num_e, num_orb, num_k):
    """L_0[g] = sum over occupied i of <i| L^g |i>, summed across all k."""
    mask = np.array(num_k * (num_e * [True] + (num_orb - num_e) * [False]))
    return np.sum(h2[mask, mask], axis=0)


def gen_A_e(h2):
    return (h2 + np.einsum("ijG->jiG", h2.conj())) / 2


def gen_A_o(h2):
    return (h2 - np.einsum("ijG->jiG", h2.conj())) * 1j / 2


def build_default_q_list(num_k):
    """Generate Q_list using the heuristic abs(K1-K2)==Q-1.

    Used only when no Q_list.npy is supplied.
    """
    ql = []
    for K1 in range(1, num_k + 1):
        for K2 in range(1, num_k + 1):
            for Q in range(1, num_k + 1):
                if abs(K1 - K2) == Q - 1:
                    ql.append([K1, K2, Q])
    return np.array(ql, dtype=np.int64)


# -------------------------------------------------------------------------
#  Momentum-compressed H2 storage:
#       dense:       (nb*nk, nb*nk, ng*nk)
#       compressed:  (nb*nk, nb,    ng*nk)   — k2 implicit (= k2_map[k1, Q])
#
#  Saves nk-fold disk + RAM on the static H2 array. At setup time the
#  compressed array is expanded to dense so the existing contractions are
#  reused — see vafpy_v4 for a fully gather-based runtime that avoids
#  ever materialising the dense form.
# -------------------------------------------------------------------------
def build_k2_map(q_list, num_k):
    """k2_map[k1, Q] = k2 (zero-indexed) such that (k1+1, k2+1, Q+1) ∈ Q-list.

    Requires the Q-list to give a *unique* k2 for each (k1, Q) — i.e. true
    canonical momentum conservation. Raises ValueError otherwise.
    """
    k2_map = -np.ones((num_k, num_k), dtype=np.int64)
    seen = {}
    for row in q_list:
        K1, K2, Q = int(row[0]), int(row[1]), int(row[2])
        key = (K1 - 1, Q - 1)
        if key in seen and seen[key] != K2 - 1:
            raise ValueError(
                f"Q-list maps (k1={K1}, Q={Q}) to multiple k2 values "
                f"({seen[key] + 1} and {K2}). Momentum-compressed H2 "
                "storage requires a unique k2 for each (k1, Q)."
            )
        seen[key] = K2 - 1
        k2_map[K1 - 1, Q - 1] = K2 - 1
    return k2_map


def compress_h2(h2_dense, q_list, num_k, num_orb, num_g_total):
    """(nb*nk, nb*nk, num_g_total) → (nb*nk, nb, num_g_total)."""
    if h2_dense.shape[1] == num_orb:
        return h2_dense
    if num_g_total % num_k != 0:
        raise ValueError(
            f"num_g_total ({num_g_total}) must be divisible by num_k ({num_k})."
        )
    ng_per_q = num_g_total // num_k
    k2_map = build_k2_map(q_list, num_k)
    h2_c = np.zeros(
        (num_orb * num_k, num_orb, num_g_total), dtype=h2_dense.dtype
    )
    for k1 in range(num_k):
        for Q in range(num_k):
            k2 = k2_map[k1, Q]
            if k2 < 0:
                continue
            h2_c[k1 * num_orb:(k1 + 1) * num_orb, :,
                 Q * ng_per_q:(Q + 1) * ng_per_q] = h2_dense[
                k1 * num_orb:(k1 + 1) * num_orb,
                k2 * num_orb:(k2 + 1) * num_orb,
                Q * ng_per_q:(Q + 1) * ng_per_q,
            ]
    return h2_c


def expand_h2(h2_c, q_list, num_k, num_orb, num_g_total):
    """(nb*nk, nb, num_g_total) → (nb*nk, nb*nk, num_g_total)."""
    if h2_c.shape[1] == num_orb * num_k:
        return h2_c
    if num_g_total % num_k != 0:
        raise ValueError(
            f"num_g_total ({num_g_total}) must be divisible by num_k ({num_k})."
        )
    ng_per_q = num_g_total // num_k
    k2_map = build_k2_map(q_list, num_k)
    h2_d = np.zeros(
        (num_orb * num_k, num_orb * num_k, num_g_total), dtype=h2_c.dtype
    )
    for k1 in range(num_k):
        for Q in range(num_k):
            k2 = k2_map[k1, Q]
            if k2 < 0:
                continue
            h2_d[k1 * num_orb:(k1 + 1) * num_orb,
                 k2 * num_orb:(k2 + 1) * num_orb,
                 Q * ng_per_q:(Q + 1) * ng_per_q] = h2_c[
                k1 * num_orb:(k1 + 1) * num_orb, :,
                Q * ng_per_q:(Q + 1) * ng_per_q,
            ]
    return h2_d


def is_h2_compressed(h2_shape, num_orb, num_k):
    """True if the array is stored compressed (nb*nk, nb, ng*nk).

    For num_k == 1 the two forms coincide; treated as dense to preserve
    backwards-compatibility with existing single-k data files.
    """
    if len(h2_shape) != 3:
        return False
    return h2_shape[1] == num_orb and num_k > 1


def build_k1_map(q_list, num_k):
    """Inverse of k2_map: k1_map[k2, Q] = k1 with (k1+1, k2+1, Q+1) ∈ Q-list.

    Used for the daggered factor (H2†) when keeping H2 compressed.
    """
    k1_map = -np.ones((num_k, num_k), dtype=np.int64)
    seen = {}
    for row in q_list:
        K1, K2, Q = int(row[0]), int(row[1]), int(row[2])
        key = (K2 - 1, Q - 1)
        if key in seen and seen[key] != K1 - 1:
            raise ValueError(
                f"Q-list maps (k2={K2}, Q={Q}) to multiple k1 values; "
                "compressed runtime requires a unique k1 for each (k2, Q)."
            )
        seen[key] = K1 - 1
        k1_map[K2 - 1, Q - 1] = K1 - 1
    return k1_map


def save_h2_compressed(h2_dense, q_list, num_k, num_orb, num_g_total, filename):
    """Convenience: compress a dense H2 array and save it to disk."""
    h2_c = compress_h2(h2_dense, q_list, num_k, num_orb, num_g_total)
    np.save(filename, h2_c)
    return h2_c


# =========================================================================
#  Backend abstraction
# =========================================================================
class Backend:
    """Thin wrapper that proxies missing attributes to the wrapped module."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)


class NumpyBackend(Backend):
    def __init__(self, seed):
        super().__init__(np)
        self.block_diag = scipy.linalg.block_diag
        self.expm = scipy.linalg.expm
        self._rng = np.random.default_rng(seed)

    def random_normal(self, shape, dtype):
        return self._rng.standard_normal(shape).astype(dtype)

    def random_uniform(self, shape=(), dtype=np.float64):
        return self._rng.uniform(size=shape).astype(dtype)

    def random_uniform_scalar(self, dtype=np.float64):
        return self._rng.uniform(0, 1, size=()).astype(dtype)

    def qr(self, matrix):
        return np.linalg.qr(matrix)

    def to_numpy(self, arr):
        return np.asarray(arr)


class JaxBackend(Backend):
    def __init__(self, seed):
        jax, jnp = _try_import_jax()
        super().__init__(jnp)
        self._jax = jax
        self._jnp = jnp
        self._key = jax.random.key(seed)

    def block_diag(self, *matrices):
        # scipy on host side, then ship to device — keeps things simple.
        on_host = [np.asarray(m) for m in matrices]
        return self._jnp.array(scipy.linalg.block_diag(*on_host))

    def expm(self, matrix):
        return self._jnp.array(scipy.linalg.expm(np.asarray(matrix)))

    def random_normal(self, shape, dtype):
        self._key, sub = self._jax.random.split(self._key)
        return self._jax.random.normal(sub, shape, dtype)

    def random_uniform(self, shape=(), dtype=None):
        if dtype is None:
            dtype = self._jnp.float32
        self._key, sub = self._jax.random.split(self._key)
        return self._jax.random.uniform(sub, shape=shape, dtype=dtype)

    def random_uniform_scalar(self, dtype=None):
        return self.random_uniform((), dtype=dtype)

    def qr(self, matrix):
        return self._jnp.linalg.qr(matrix)

    def to_numpy(self, arr):
        return np.asarray(arr)


class CupyBackend(Backend):
    def __init__(self, seed):
        cp = _try_import_cupy()
        super().__init__(cp)
        self._cp = cp
        self._rng = cp.random.default_rng(seed)

    def block_diag(self, *matrices):
        on_host = [m.get() if hasattr(m, "get") else np.asarray(m) for m in matrices]
        return self._cp.array(scipy.linalg.block_diag(*on_host))

    def expm(self, matrix):
        host = matrix.get() if hasattr(matrix, "get") else np.asarray(matrix)
        return self._cp.array(scipy.linalg.expm(host))

    def random_normal(self, shape, dtype):
        return self._rng.standard_normal(shape).astype(dtype)

    def random_uniform(self, shape=(), dtype=None):
        if dtype is None:
            dtype = self._cp.float64
        return self._rng.uniform(size=shape).astype(dtype)

    def random_uniform_scalar(self, dtype=None):
        return self.random_uniform((), dtype=dtype)

    def qr(self, matrix):
        return self._cp.linalg.qr(matrix)

    def to_numpy(self, arr):
        return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def make_backend(name, seed):
    name = name.lower()
    if name == "numpy":
        return NumpyBackend(seed)
    if name == "jax":
        return JaxBackend(seed)
    if name == "cupy":
        return CupyBackend(seed)
    raise NotImplementedError(f"Backend '{name}' not implemented.")


# =========================================================================
#  Configuration / data classes
# =========================================================================
@dataclass
class Configuration:
    num_walkers: int
    num_kpoint: int
    num_orbital: int
    num_electron: int
    num_g: int
    singularity: float
    propagator: str
    order_propagation: int
    timestep: float
    comm: MPI.Comm
    precision: str
    backend: Backend

    @property
    def float_type(self):
        if self.precision == "Single":
            return self.backend.single
        if self.precision == "Double":
            return self.backend.double
        raise NotImplementedError(f"precision '{self.precision}' unknown")

    @property
    def complex_type(self):
        if self.precision == "Single":
            return self.backend.csingle
        if self.precision == "Double":
            return self.backend.cdouble
        raise NotImplementedError(f"precision '{self.precision}' unknown")


@dataclass
class Walkers:
    slater_det: object
    weights: object


@dataclass
class Hamiltonian:
    """v4: keeps H2 in the compressed (nb*nk, nb, ng*nk) form at runtime.

    The energy expressions (Hartree, Exchange) and the per-step auxiliary
    field application are evaluated via a Python loop over Q with NumPy
    advanced-indexed gathers — so the dense (nb*nk, nb*nk, w) propagator
    matrix is never materialised. Setup-time mean-field corrections still
    use a temporary dense expansion of H2 for simplicity (one-shot cost).
    """
    one_body: object  # dense (nb*nk, nb*nk)
    two_body: object  # compressed (nb*nk, nb, ng*nk)
    H_zero: complex = 0.0
    q_list: object = None
    test_random_field: object = None

    def setup_energy_expressions(self, config, trial_det):
        nb = config.num_orbital
        nk = config.num_kpoint
        ne = config.num_electron
        nb_k = nb * nk
        ne_k = ne * nk
        ng_t = config.num_g
        if ng_t % nk != 0:
            raise ValueError(
                f"num_g ({ng_t}) must be divisible by num_kpoint ({nk})."
            )
        ng_per_q = ng_t // nk

        # Q-list and the (k1, Q) → k2 and (k2, Q) → k1 lookup tables.
        if self.q_list is None:
            self.q_list = build_default_q_list(nk)
        ql = self.q_list
        self._k2_map = build_k2_map(ql, nk)
        self._k1_map = build_k1_map(ql, nk)
        # Precompute the (Q, k_p, r_orb) -> row_index_in_theta lookups so the
        # per-Q gather is one advanced-index step (no reshape + diagonal dance).
        if nk > 1:
            r_orb = np.arange(nb)
            # row_idx[Q, k_p, r_orb] = k2_map[k_p, Q] * nb + r_orb
            self._k2_row_idx = (
                self._k2_map.T[:, :, None] * nb + r_orb[None, None, :]
            ).astype(np.intp)
            self._k1_row_idx = (
                self._k1_map.T[:, :, None] * nb + r_orb[None, None, :]
            ).astype(np.intp)
            # col_idx[k_p, i_local] = k_p * ne + i_local
            self._col_idx = (
                np.arange(nk)[:, None] * ne + np.arange(ne)[None, :]
            ).astype(np.intp)
            # walker column index: k_p * ne + i_local (same as col_idx) — but
            # for apply_h2 we want the FULL ne_k columns, so use a different alias.
            self._walker_col_full = np.arange(ne * nk).astype(np.intp)

        # Pull host copies of the inputs.
        trial_host = config.backend.to_numpy(trial_det)
        trial_single = trial_host[:nb, :ne]
        h1_host = config.backend.to_numpy(self.one_body)
        h2_in = config.backend.to_numpy(self.two_body)
        # Accept either dense or compressed input.
        if is_h2_compressed(h2_in.shape, nb, nk):
            h2_c_host = h2_in
            h2_dense_host = expand_h2(h2_c_host, ql, nk, nb, ng_t)
        else:
            h2_dense_host = h2_in
            h2_c_host = compress_h2(h2_in, ql, nk, nb, ng_t) if nk > 1 else h2_in
        h2_dag_dense = np.einsum("prG->rpG", h2_dense_host.conj())

        # --- one-time setup uses the dense form (reuses v3 helpers) -----
        h_mf = H_1_mf(trial_single, trial_host, h2_dense_host, h2_dag_dense,
                      ql, h1_host, nk, nb, ne)
        h_sic = -contract("ijG, kjG -> ik", h2_dense_host, h2_dag_dense) / 2
        h1_total = h_mf + h_sic

        L_0 = mean_field_diag(h2_dense_host, ne, nb, nk)
        self.H_zero = 2 * np.einsum("g,g->", L_0, L_0.conj()) / (2 * ne)

        h2_mf_dense = A_af_MF_sub(trial_single, trial_host, h2_dense_host,
                                  ql, nk, nb, ne)
        # Re-compress the mean-field-subtracted H2 (one block per (k1, Q));
        # we will recover h2_e / h2_o on the fly via the identity
        #   h2_e·x_e + h2_o·x_o = h2_mf·z₊ + h2_mf†·z₋
        # with z₊ = (x_e + i·x_o)/2, z₋ = (x_e − i·x_o)/2.
        if nk > 1:
            h2_mf_c = compress_h2(h2_mf_dense, ql, nk, nb, ng_t)
        else:
            h2_mf_c = h2_mf_dense
        self._h2_mf_c = h2_mf_c
        self._h2_mf_c_5d = h2_mf_c.reshape(nk, nb, nb, nk, ng_per_q) if nk > 1 \
            else h2_mf_c.reshape(1, nb, nb, 1, ng_t)

        # Compressed α and α_T (from the RAW H2) for hartree/exchange.
        alpha_c = self._compute_alpha_c(h2_c_host, trial_single, nk, nb, ne, ng_t)
        alpha_T_c = self._compute_alpha_T_c(
            h2_c_host, trial_single, ql, nk, nb, ne, ng_t
        )
        self._alpha_c_5d = alpha_c.reshape(nk, ne, nb, nk, ng_per_q) if nk > 1 \
            else alpha_c.reshape(1, ne, nb, 1, ng_t)
        self._alpha_T_c_5d = alpha_T_c.reshape(nk, ne, nb, nk, ng_per_q) if nk > 1 \
            else alpha_T_c.reshape(1, ne, nb, 1, ng_t)

        # α_mf_c / α_mf_T_c — built from h2_mf_c. Force-bias is
        #   fb_e = (fb_mf + fb_mf_T)/2,  fb_o = (fb_mf − fb_mf_T)*1j/2
        alpha_mf_c = self._compute_alpha_c(h2_mf_c, trial_single, nk, nb, ne, ng_t)
        alpha_mf_T_c = self._compute_alpha_T_c(
            h2_mf_c, trial_single, ql, nk, nb, ne, ng_t
        )
        self._alpha_mf_c_5d = alpha_mf_c.reshape(nk, ne, nb, nk, ng_per_q) if nk > 1 \
            else alpha_mf_c.reshape(1, ne, nb, 1, ng_t)
        self._alpha_mf_T_c_5d = alpha_mf_T_c.reshape(nk, ne, nb, nk, ng_per_q) if nk > 1 \
            else alpha_mf_T_c.reshape(1, ne, nb, 1, ng_t)

        # one-body expression (h1_trial @ theta) — small, keep dense
        h1_trial = contract("pi, pq -> iq", trial_host, h1_host)
        self._h1_trial = config.backend.array(h1_trial, dtype=config.complex_type)

        h1_total_be = config.backend.array(h1_total, dtype=config.complex_type)
        self._h1 = -h1_total_be * config.timestep
        self._exp_h1 = config.backend.expm(-h1_total_be * config.timestep)
        self._exp_h1_half = config.backend.expm(-0.5 * h1_total_be * config.timestep)

        self._sqrt_tau = config.backend.sqrt(
            config.backend.array(config.timestep)
        ).astype(config.float_type)

        self._nb = nb
        self._nk = nk
        self._ne = ne
        self._ng_t = ng_t
        self._ng_per_q = ng_per_q
        self._singularity_correction = ne * nk * config.singularity

    # ------------------------------------------------------------------
    #  Compressed-form precomputations
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_alpha_c(h2_c, trial_single, nk, nb, ne, ng_t):
        """alpha_c[k_i*ne+i_local, r, gQ] = Σ_p trial_single[p, i_local] * h2_c[k_i*nb+p, r, gQ]."""
        if nk == 1:
            return contract("pi, prg -> irg", trial_single, h2_c)
        ng_per_q = ng_t // nk
        h2_5d = h2_c.reshape(nk, nb, nb, nk, ng_per_q)
        alpha_5d = np.einsum("pi, kprQg -> kirQg", trial_single, h2_5d)
        return alpha_5d.reshape(ne * nk, nb, ng_t)

    @staticmethod
    def _compute_alpha_T_c(h2_c, trial_single, q_list, nk, nb, ne, ng_t):
        """alpha_T_c[k_j*ne+j_local, p_orb, gQ] = h2_c[k1_map[k_j, Q]*nb+p_orb, j_local, gQ].conj()."""
        if nk == 1:
            h2_dag = np.einsum("prG->rpG", h2_c.conj())
            return contract("pi, prg -> irg", trial_single, h2_dag)
        ng_per_q = ng_t // nk
        k1_map = build_k1_map(q_list, nk)
        h2_5d = h2_c.reshape(nk, nb, nb, nk, ng_per_q)
        alpha_T_5d = np.zeros((nk, ne, nb, nk, ng_per_q), dtype=h2_c.dtype)
        for Q in range(nk):
            # h2_5d[k1_map[k_j, Q], p_orb, j_local, Q, g]  (only first `ne` j's)
            gathered = h2_5d[k1_map[:, Q], :, :ne, Q, :]   # (k_j, p_orb, j_local, g)
            # alpha_T_5d[k_j, j_local, p_orb, Q, g] = gathered[k_j, p_orb, j_local, g].conj()
            alpha_T_5d[:, :, :, Q, :] = gathered.transpose(0, 2, 1, 3).conj()
        return alpha_T_5d.reshape(ne * nk, nb, ng_t)

    # ------------------------------------------------------------------
    #  Energy components (compressed; per-Q gather)
    # ------------------------------------------------------------------
    def compute_one_body(self, theta):
        return 2 * contract("ip, wpi -> w", self._h1_trial, theta)

    def _gather_theta_diag(self, theta, row_idx_Q):
        """Pick θ[w, row_idx_Q[k_p, r], col_idx[k_p, i]] → (w, k_p, r, i).

        row_idx_Q is (nk, nb); self._col_idx is (nk, ne).
        Replaces the reshape→fancy-diag→transpose chain with a single
        advanced-index op.
        """
        return theta[
            :,
            row_idx_Q[:, :, None],         # (nk, nb, 1)
            self._col_idx[:, None, :],     # (nk, 1, ne)
        ]                                  # → (w, nk, nb, ne)

    def compute_hartree(self, theta):
        nb, nk, ne = self._nb, self._nk, self._ne
        if nk == 1:
            alpha_c = self._alpha_c_5d.reshape(ne, nb, -1)
            alpha_T_c = self._alpha_T_c_5d.reshape(ne, nb, -1)
            fb = contract("wri, irg -> gw", theta, alpha_c)
            fb_T = contract("wri, irg -> gw", theta, alpha_T_c)
            return 2 * contract("gw, gw -> w", fb, fb_T)
        nw = theta.shape[0]
        e_h = np.zeros(nw, dtype=theta.dtype)
        for Q in range(nk):
            θ_k2_diag = self._gather_theta_diag(theta, self._k2_row_idx[Q])
            θ_k1_diag = self._gather_theta_diag(theta, self._k1_row_idx[Q])
            α_Q = self._alpha_c_5d[:, :, :, Q, :]
            α_T_Q = self._alpha_T_c_5d[:, :, :, Q, :]
            fb_Q = np.einsum("kirg, wkri -> wg", α_Q, θ_k2_diag)
            fb_T_Q = np.einsum("kjrg, wkrj -> wg", α_T_Q, θ_k1_diag)
            e_h = e_h + np.einsum("wg, wg -> w", fb_Q, fb_T_Q)
        return 2 * e_h

    def compute_exchange(self, theta):
        nb, nk, ne = self._nb, self._nk, self._ne
        if nk == 1:
            alpha_c = self._alpha_c_5d.reshape(ne, nb, -1)
            alpha_T_c = self._alpha_T_c_5d.reshape(ne, nb, -1)
            return -contract("wri, jrg, wpj, ipg -> w",
                             theta, alpha_T_c, theta, alpha_c)
        nw = theta.shape[0]
        ne_k = ne * nk
        # For exchange we still need the 5-D θ tensors so we can mix two
        # different k-axes (one for the α side, one for the α_T side).
        theta_4d = theta.reshape(nw, nk, nb, ne_k)
        e_x = np.zeros(nw, dtype=theta.dtype)
        for Q in range(nk):
            θ_k1 = theta_4d[:, self._k1_map[:, Q], :, :].reshape(
                nw, nk, nb, nk, ne
            )  # (w, k_j, r, k_i, i_local)
            θ_k2 = theta_4d[:, self._k2_map[:, Q], :, :].reshape(
                nw, nk, nb, nk, ne
            )  # (w, k_i, p, k_j, j_local)
            α_Q = self._alpha_c_5d[:, :, :, Q, :]
            α_T_Q = self._alpha_T_c_5d[:, :, :, Q, :]
            e_x_Q = np.einsum(
                "wARBX, AYRg, wBPAY, BXPg -> w",
                θ_k1, α_T_Q, θ_k2, α_Q,
            )
            e_x = e_x + e_x_Q
        return -e_x

    @staticmethod
    def _zeros_w(reference, nw):
        # match dtype/backend of theta
        return np.zeros(nw, dtype=reference.dtype) if hasattr(reference, "dtype") else 0

    # ------------------------------------------------------------------
    #  Auxiliary field (compressed; never materialises the dense matrix)
    # ------------------------------------------------------------------
    def create_random_field(self, config):
        if self.test_random_field is None:
            return config.backend.random_normal(
                (2 * config.num_g, config.num_walkers), config.float_type
            )
        return self.test_random_field

    def create_auxiliary_field(self, config, theta):
        """Return (x_e_eff, x_o_eff, importance).

        x_eff = random_field − force_bias is what gets contracted with H2 in
        apply_propagator. We never form the dense (nb*nk, nb*nk, w) matrix.
        """
        nk, nb, ne = self._nk, self._nb, self._ne
        ng_t = self._ng_t
        random_field = self.create_random_field(config)        # (2*ng_t, w)
        force_bias_eo = self._compute_force_bias_compressed(theta)  # (2*ng_t, w)
        force_bias_eo = -2j * self._sqrt_tau * force_bias_eo
        force_bias_eo = config.backend.where(
            abs(force_bias_eo) > 1, 0.0, force_bias_eo
        )
        arg = contract(
            "gw, gw -> w",
            random_field - 0.5 * force_bias_eo,
            force_bias_eo,
        )
        shifted = random_field - force_bias_eo
        # split into the e- and o-halves
        x_e = shifted[:ng_t]
        x_o = shifted[ng_t:]
        return x_e, x_o, config.backend.exp(arg)

    def _compute_force_bias_compressed(self, theta):
        """fb[gQ_full, w] over the concatenated (e, o) g-axis.

        Uses the decomposition α_e = (α_mf + α_mf_T)/2,
        α_o = (α_mf − α_mf_T)*1j/2 so we only need h2_mf_c-derived alphas.
        """
        nb, nk, ne = self._nb, self._nk, self._ne
        ng_t = self._ng_t
        nw = theta.shape[0]
        if nk == 1:
            alpha_mf = self._alpha_mf_c_5d.reshape(ne, nb, -1)
            alpha_mf_T = self._alpha_mf_T_c_5d.reshape(ne, nb, -1)
            fb_mf = contract("wri, irg -> gw", theta, alpha_mf)
            fb_mf_T = contract("wri, irg -> gw", theta, alpha_mf_T)
            fb_e = (fb_mf + fb_mf_T) / 2
            fb_o = (fb_mf - fb_mf_T) * 1j / 2
            return np.concatenate((fb_e, fb_o), axis=0)

        fb_mf = np.zeros((ng_t, nw), dtype=theta.dtype)
        fb_mf_T = np.zeros((ng_t, nw), dtype=theta.dtype)
        for Q in range(nk):
            θ_k2_diag = self._gather_theta_diag(theta, self._k2_row_idx[Q])
            θ_k1_diag = self._gather_theta_diag(theta, self._k1_row_idx[Q])
            α_mf_Q = self._alpha_mf_c_5d[:, :, :, Q, :]
            α_mf_T_Q = self._alpha_mf_T_c_5d[:, :, :, Q, :]
            fb_mf[Q * self._ng_per_q:(Q + 1) * self._ng_per_q, :] = np.einsum(
                "kirg, wkri -> gw", α_mf_Q, θ_k2_diag
            )
            fb_mf_T[Q * self._ng_per_q:(Q + 1) * self._ng_per_q, :] = np.einsum(
                "kirg, wkri -> gw", α_mf_T_Q, θ_k1_diag
            )
        fb_e = (fb_mf + fb_mf_T) / 2
        fb_o = (fb_mf - fb_mf_T) * 1j / 2
        return np.concatenate((fb_e, fb_o), axis=0)

    def apply_h2_compressed(self, slater_det, x_e, x_o):
        """Apply (1j * sqrt_tau * h2_op) to slater_det without materialising the matrix.

        Uses h2_e·x_e + h2_o·x_o = h2_mf·z₊ + h2_mf†·z₋,
        where z₊ = (x_e + 1j x_o)/2, z₋ = (x_e − 1j x_o)/2.
        """
        nb, nk, ne_k = self._nb, self._nk, self._ne * self._nk
        nw = slater_det.shape[0]
        sqrt_tau = self._sqrt_tau
        z_plus = (x_e + 1j * x_o) / 2
        z_minus = (x_e - 1j * x_o) / 2

        if nk == 1:
            field = (
                contract("prg, gw -> prw",
                         self._h2_mf_c_5d.reshape(nb, nb, -1), z_plus)
                + contract("prg, gw -> prw",
                           self._h2_mf_c_5d.reshape(nb, nb, -1).conj()
                           .transpose(1, 0, 2), z_minus)
            )
            field = 1j * sqrt_tau * field
            return contract("prw, wri -> wpi", field, slater_det)

        sd_4d = slater_det.reshape(nw, nk, nb, ne_k)
        result_4d = np.zeros_like(sd_4d)
        for Q in range(nk):
            z_plus_Q = z_plus[Q * self._ng_per_q:(Q + 1) * self._ng_per_q, :]
            z_minus_Q = z_minus[Q * self._ng_per_q:(Q + 1) * self._ng_per_q, :]
            # h2_mf path: gather walker via k2_map.
            h2_mf_Q = self._h2_mf_c_5d[:, :, :, Q, :]   # (k_p, p, r, g)
            field_mf = np.einsum("kprg, gw -> kprw", h2_mf_Q, z_plus_Q)
            sd_gathered_k2 = sd_4d[:, self._k2_map[:, Q], :, :]
            result_4d += np.einsum("kprw, wkri -> wkpi", field_mf, sd_gathered_k2)

            # h2_mf† path: gather walker via k1_map, swap+conj h2_mf_c entries.
            # h2_mf†[k_p*nb+p, k1_map[k_p,Q]*nb+r, gQ]
            #   = h2_mf[k1_map[k_p,Q]*nb+r, k_p*nb+p, gQ].conj()
            #   = h2_mf_c[k1_map[k_p,Q]*nb+r, p, gQ].conj()
            gathered = self._h2_mf_c_5d[self._k1_map[:, Q], :, :, Q, :]
            # gathered axes: (k_p, p_row_at_k1_map, r_at_k_p, g)
            # We want field_dag[k_p, p, r, w] = Σ_g gathered[k_p, r, p, g].conj() * z_minus_Q[g, w]
            field_dag = np.einsum(
                "krpg, gw -> kprw", gathered.conj(), z_minus_Q
            )
            sd_gathered_k1 = sd_4d[:, self._k1_map[:, Q], :, :]
            result_4d += np.einsum(
                "kprw, wkri -> wkpi", field_dag, sd_gathered_k1
            )

        return (1j * sqrt_tau * result_4d).reshape(slater_det.shape)

    def apply_taylor_compressed(self, slater_det, x_e, x_o, order):
        """exp(1j*sqrt_tau*h2_op) via order-N Taylor series, compressed-style."""
        result = slater_det.copy()
        addend = slater_det
        for i in range(order):
            addend = self.apply_h2_compressed(addend, x_e, x_o) / (i + 1)
            result = result + addend
        return result

    @property
    def h1(self):
        return self._h1

    @property
    def exp_h1(self):
        return self._exp_h1

    @property
    def exp_h1_half(self):
        return self._exp_h1_half


# =========================================================================
#  I/O
# =========================================================================
def obtain_H1(config, filename="H1_svd.npy"):
    """Load per-k H1 and assemble block-diagonal multi-k matrix."""
    h1_per_k = np.load(filename).astype(np.complex128)
    # Expected shape (num_orb, num_orb, num_k) — what opt/vasp produce.
    if h1_per_k.ndim == 2:
        h1_per_k = h1_per_k[:, :, None]
    h1 = reshape_H1(h1_per_k, config.num_kpoint, config.num_orbital)
    return config.backend.array(h1, dtype=config.complex_type)


def obtain_H2(config, filename="H2_zip.npy", q_list=None, keep_compressed=True):
    """Load H2 from disk.

    Accepts either of two layouts:
      * dense       — (num_orb*num_k, num_orb*num_k, num_g_total)
      * compressed  — (num_orb*num_k, num_orb,       num_g_total)

    In v4 the runtime keeps H2 in the compressed form throughout, so by
    default we compress dense inputs (rather than expanding compressed
    inputs as v3 does). Set keep_compressed=False to mimic v3 behaviour.
    """
    h2 = np.load(filename).astype(np.complex128)
    is_compressed = is_h2_compressed(
        h2.shape, config.num_orbital, config.num_kpoint
    )
    if keep_compressed:
        if not is_compressed and config.num_kpoint > 1:
            if q_list is None:
                raise ValueError(
                    "Dense H2 with num_k>1 needs q_list to be compressed."
                )
            h2 = compress_h2(
                h2, q_list, config.num_kpoint,
                config.num_orbital, config.num_g,
            )
        return config.backend.array(h2, dtype=config.complex_type)
    # legacy expand path
    if is_compressed:
        if q_list is None:
            raise ValueError(
                "Compressed H2 detected — pass q_list to obtain_H2."
            )
        h2 = expand_h2(
            h2, q_list, config.num_kpoint, config.num_orbital, config.num_g,
        )
    return config.backend.array(h2, dtype=config.complex_type)


def obtain_Q_list(config, filename="Q_list.npy"):
    """Load Q-list. Layout: rows are [K1, K2, Q]. Falls back to heuristic."""
    if not os.path.exists(filename):
        return build_default_q_list(config.num_kpoint)
    ql = np.load(filename)
    # Files coming from the legacy pipeline are (3, N) — transpose if so.
    if ql.shape[0] == 3 and ql.shape[1] != 3:
        ql = ql.T
    return ql.astype(np.int64, copy=False)


def initialize_determinant(config):
    """Block-diagonal multi-k trial determinant and initial walker copies."""
    single = config.backend.eye(
        config.num_orbital, config.num_electron, dtype=config.float_type
    )
    trial_det = config.backend.block_diag(*([single] * config.num_kpoint))
    slater_det = config.backend.array(
        config.num_walkers * [config.backend.to_numpy(trial_det)]
    ).astype(config.complex_type)
    walkers = Walkers(
        slater_det=slater_det,
        weights=config.backend.ones(config.num_walkers, dtype=config.complex_type),
    )
    return trial_det, walkers


# =========================================================================
#  Energy and propagation
# =========================================================================
def biorthogonalize(backend, trial, slater_det):
    inverse_overlap = backend.linalg.inv(trial.T @ slater_det)
    return contract("wpi, wij -> wpj", slater_det, inverse_overlap)


def project_trial(backend, trial, slater_det):
    return backend.linalg.det(trial.T @ slater_det) ** 2


def measure_energy(config, trial, walkers, hamiltonian):
    th = biorthogonalize(config.backend, trial, walkers.slater_det)
    e1 = hamiltonian.compute_one_body(th)
    eh = hamiltonian.compute_hartree(th)
    ex = hamiltonian.compute_exchange(th)
    energy = (e1 + eh + ex) / config.num_kpoint
    weighted_energy = energy @ walkers.weights
    sum_weights = config.backend.sum(walkers.weights)
    weighted_energy_global = config.comm.allreduce(weighted_energy)
    sum_weights_global = config.comm.allreduce(sum_weights)
    return (
        weighted_energy_global / sum_weights_global,
        weighted_energy_global,
        sum_weights_global,
    )


def measure_components(config, trial, walkers, hamiltonian):
    """Returns (E_one, Hartree, Exchange) — useful for verification."""
    th = biorthogonalize(config.backend, trial, walkers.slater_det)
    e1 = hamiltonian.compute_one_body(th)
    eh = hamiltonian.compute_hartree(th)
    ex = hamiltonian.compute_exchange(th)
    w = walkers.weights
    sumw = config.backend.sum(w)
    return (
        (e1 @ w) / sumw / config.num_kpoint,
        (eh @ w) / sumw / config.num_kpoint,
        (ex @ w) / sumw / config.num_kpoint,
    )


def apply_taylor(config, matrix, slater_det):
    result = slater_det.copy()
    addend = slater_det
    for i in range(config.order_propagation):
        addend = contract("pqw, wqi -> wpi", matrix, addend) / (i + 1)
        result += addend
    return result


def propagate_walkers(config, trial, walkers, hamiltonian, h_0, e_0):
    """One imaginary-time step of all walkers.

    v4: applies h2_op directly to the walker via per-Q gather; the dense
    (nb*nk, nb*nk, w) propagator matrix is NEVER materialised.
    """
    new_walkers = Walkers(
        config.backend.zeros_like(walkers.slater_det),
        config.backend.zeros_like(walkers.weights),
    )
    th = biorthogonalize(config.backend, trial, walkers.slater_det)
    x_e, x_o, importance = hamiltonian.create_auxiliary_field(config, th)
    num_rare_event = 0
    order = config.order_propagation

    if config.propagator == "S2":
        half = hamiltonian.exp_h1_half @ walkers.slater_det
        full_h2 = hamiltonian.apply_taylor_compressed(half, x_e, x_o, order)
        half = hamiltonian.exp_h1_half @ full_h2
        new_walkers.slater_det = h_0 * half
    elif config.propagator == "S1":
        full_h2 = hamiltonian.apply_taylor_compressed(
            walkers.slater_det, x_e, x_o, order
        )
        full_h1 = hamiltonian.exp_h1 @ full_h2
        new_walkers.slater_det = h_0 * full_h1
    elif config.propagator == "Taylor":
        raise NotImplementedError(
            "Combined exp(h1+h2) Taylor isn't supported in v4 — use S1 or S2."
        )
    else:
        raise ValueError(f"Unknown propagator '{config.propagator}'")

    new_overlap = project_trial(config.backend, trial, new_walkers.slater_det)
    old_overlap = project_trial(config.backend, trial, walkers.slater_det)
    overlap_ratio = new_overlap / old_overlap
    cos_alpha = config.backend.cos(config.backend.angle(overlap_ratio))
    factor = abs(overlap_ratio * importance * config.backend.exp(config.timestep * e_0))
    factor = config.backend.where(factor < 10, factor, 0)
    num_rare_event += int(config.backend.sum(factor == 0))
    new_walkers.weights = abs(factor) * config.backend.maximum(0, cos_alpha) * walkers.weights
    return new_walkers, num_rare_event


# =========================================================================
#  Re-orthogonalisation, weight rebalancing
# =========================================================================
def reortho_qr(config, walker_matrix):
    Q, _ = config.backend.qr(walker_matrix)
    return Q


def init_walkers_weights(config, n_walkers):
    return config.backend.ones(n_walkers, dtype=config.complex_type)


def rebalance_comb(config, weights):
    """Systematic resampling on a single rank; returns new walker indices."""
    w = config.backend.to_numpy(weights).real
    N = len(w)
    c = np.cumsum(w)
    W = c[-1]
    r = float(config.backend.random_uniform_scalar()) * (W / N)
    U = r + np.arange(N) * (W / N)
    new_indices = np.searchsorted(c, U, side="left")
    new_indices[new_indices >= N] = N - 1
    return new_indices.astype(np.int64)


def rebalance_global(comm, walkers_mats_up, walkers_weights, config):
    """Global rebalance across MPI ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_n, n_orb, n_elec = walkers_mats_up.shape
    total_n = local_n * size

    all_weights = None
    if rank == 0:
        all_weights = np.empty(total_n, dtype=np.complex128)
    comm.Gather(walkers_weights, all_weights, root=0)

    instances = None
    if rank == 0:
        norm = total_n / np.sum(all_weights.real)
        norm_w = all_weights.real * norm
        bias = -float(config.backend.random_uniform_scalar())
        instances = np.zeros(total_n, dtype=int)
        cum = bias
        prev = 0
        for i in range(total_n):
            cum += norm_w[i]
            cur = int(np.ceil(cum))
            instances[i] = cur - prev
            prev = cur

    if rank != 0:
        instances = np.empty(total_n, dtype=int)
    comm.Bcast(instances, root=0)

    map_indices = np.empty(int(np.sum(instances)), dtype=int)
    count = 0
    for idx, n in enumerate(instances):
        for _ in range(n):
            map_indices[count] = idx
            count += 1
    assert len(map_indices) == total_n

    all_mats = None
    if rank == 0:
        all_mats = np.empty((total_n, n_orb, n_elec), dtype=np.complex128)
    comm.Gather(walkers_mats_up, all_mats, root=0)

    resampled = all_mats[map_indices] if rank == 0 else None
    new_mats = np.empty((local_n, n_orb, n_elec), dtype=np.complex128)
    comm.Scatter(resampled, new_mats, root=0)
    new_weights = np.ones(local_n, dtype=np.complex128)
    return new_mats, new_weights


# =========================================================================
#  Analysis
# =========================================================================
def blockAverage(datastream, block_divisor):
    Nobs = len(datastream)
    minBlockSize = 1
    maxBlockSize = int(Nobs / block_divisor)
    NumBlocks = maxBlockSize - minBlockSize
    blockMean = np.zeros(NumBlocks)
    blockVar = np.zeros(NumBlocks)
    ctr = 0
    for blockSize in range(minBlockSize, maxBlockSize):
        Nblock = int(Nobs / blockSize)
        obsProp = np.zeros(Nblock)
        for i in range(1, Nblock + 1):
            ibeg = (i - 1) * blockSize
            iend = ibeg + blockSize
            obsProp[i - 1] = np.mean(datastream[ibeg:iend])
        blockMean[ctr] = np.mean(obsProp)
        blockVar[ctr] = np.var(obsProp) / (Nblock - 1)
        ctr += 1
    v = np.arange(minBlockSize, maxBlockSize)
    return v, blockVar, blockMean
