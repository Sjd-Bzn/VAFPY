"""
vafpy_v3: unified GPU/CPU AFQMC with k-point support.

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
        # Ampere/Ada GPUs use TF32 for matmuls by default; force full float32.
        jax.config.update("jax_default_matmul_precision", "float32")
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
    """Holds one- and two-body terms plus precomputed contract expressions."""
    one_body: object  # (num_orb*num_k, num_orb*num_k)
    two_body: object  # (num_orb*num_k, num_orb*num_k, num_g_total)
    H_zero: complex = 0.0
    q_list: object = None
    test_random_field: object = None

    def setup_energy_expressions(self, config, trial_det):
        nb_k = config.num_orbital * config.num_kpoint
        ne_k = config.num_electron * config.num_kpoint
        shape_theta = (config.num_walkers, nb_k, ne_k)

        h1_trial = contract("pi, pq -> iq", trial_det, self.one_body)
        alpha = contract("pi, prg -> irg", trial_det, self.two_body)
        alpha_T = contract("pi, rpg -> irg", trial_det, self.two_body.conj())

        self._one_body_expression = contract_expression(
            "ip, wpi -> w", h1_trial, shape_theta,
            constants=[0], optimize="greedy",
        )
        args = (shape_theta, alpha, shape_theta, alpha_T)
        kwargs = {"constants": [1, 3], "optimize": "greedy"}
        self._hartree_expression = contract_expression(
            "wri, irg, wpj, jpg -> w", *args, **kwargs
        )
        self._exchange_expression = contract_expression(
            "wri, jrg, wpj, ipg -> w", *args, **kwargs
        )
        self._singularity_correction = config.singularity * config.num_kpoint  # FSG per unit cell; ×num_k cancels the /num_k in measure_components

        # Build / use Q_list for mean-field subtraction.
        if self.q_list is None:
            self.q_list = build_default_q_list(config.num_kpoint)
        ql = self.q_list

        # Mean-field corrected one-body Hamiltonian (computed on host).
        # avg_A_Q / get_alpha_k1_k2 expect the *single-k* trial; the multi-k
        # trial is the block-diagonal extension. Extract the first k-block.
        h2_host = config.backend.to_numpy(self.two_body)
        h1_host = config.backend.to_numpy(self.one_body)
        trial_host = config.backend.to_numpy(trial_det)
        trial_single = trial_host[:config.num_orbital, :config.num_electron]
        h2_dag = np.einsum("prG->rpG", h2_host.conj())

        h_mf = H_1_mf(trial_single, trial_host, h2_host, h2_dag, ql,
                      h1_host, config.num_kpoint,
                      config.num_orbital, config.num_electron)
        # LOCAL FIX: the exported L is unscaled; the physical interaction is V = (1/nk) sum L L^dagger
        # (verified by MP2: supercell == primitive only with 1/nk). Quadratic terms get 1/nk.
        nk = config.num_kpoint
        h_sic = -contract("ijG, jkG -> ik", h2_host, h2_dag) / (2 * nk)
        h1_total = h1_host + (h_mf - h1_host) / nk + h_sic   # H_1_mf returns h1 + change/2

        # H_zero: mean-field constant so that the full determinant scales as exp(tau*E_H_total):
        # E_H_total = 2|L0|^2/nk and the determinant has num_electron*num_kpoint columns => divide by 2*ne*nk^2 (LOCAL FIX)
        L_0 = mean_field_diag(h2_host, config.num_electron,
                              config.num_orbital, config.num_kpoint)
        self.H_zero = 2 * np.einsum("g,g->", L_0, L_0.conj()) / (2 * config.num_electron * config.num_kpoint**2)

        # Two-body after mean-field subtraction, split into Hermitian/anti-Hermitian.
        h2_mf = A_af_MF_sub(trial_single, trial_host, h2_host, ql,
                            config.num_kpoint, config.num_orbital, config.num_electron)
        two_body_e = gen_A_e(h2_mf)
        two_body_o = gen_A_o(h2_mf)
        two_body_eo = np.concatenate((two_body_e, two_body_o), axis=-1) / np.sqrt(nk)   # LOCAL FIX: L -> L/sqrt(nk) in the propagator

        # Ship the propagator pieces back to the backend.
        h1_total_be = config.backend.array(h1_total, dtype=config.complex_type)
        self._h1 = -h1_total_be * config.timestep
        self._exp_h1 = config.backend.expm(-h1_total_be * config.timestep)
        self._exp_h1_half = config.backend.expm(-0.5 * h1_total_be * config.timestep)

        two_body_eo_be = config.backend.array(two_body_eo, dtype=config.complex_type)
        alpha_eo = contract("pi, prg -> irg",
                            config.backend.to_numpy(trial_det),
                            config.backend.to_numpy(two_body_eo_be))
        alpha_eo_be = config.backend.array(alpha_eo, dtype=config.complex_type)

        self._sqrt_tau = config.backend.sqrt(
            config.backend.array(config.timestep)
        ).astype(config.float_type)
        self._force_bias_expression = contract_expression(
            "wri, irg -> gw", shape_theta, alpha_eo_be,
            constants=[1], optimize="greedy",
        )
        self._auxiliary_field = contract_expression(
            "ijg, gw -> ijw", two_body_eo_be,
            (2 * config.num_g, config.num_walkers),
            constants=[0], optimize="greedy",
        )

    def compute_one_body(self, theta):
        return 2 * self._one_body_expression(theta)

    def compute_hartree(self, theta):
        return 2 * self._hartree_expression(theta, theta)

    def compute_exchange(self, theta):
        return -self._exchange_expression(theta, theta) + self._singularity_correction

    def create_random_field(self, config):
        if self.test_random_field is None:
            return config.backend.random_normal(
                (2 * config.num_g, config.num_walkers), config.float_type
            )
        return self.test_random_field

    def create_auxiliary_field(self, config, theta):
        random_field = self.create_random_field(config)
        force_bias = -2j * self._sqrt_tau * self._force_bias_expression(theta)
        # boundary condition for rare events
        force_bias = config.backend.where(abs(force_bias) > 1, 0.0, force_bias)
        arg = contract(
            "gw, gw -> w", random_field - 0.5 * force_bias, force_bias
        )
        field = 1j * self._sqrt_tau * self._auxiliary_field(random_field - force_bias)
        return field, config.backend.exp(arg)

    @property
    def h1(self):
        return self._h1

    @property
    def exp_h1(self):
        return self._exp_h1

    @property
    def exp_h1_half(self):
        return self._exp_h1_half


@dataclass
class HamiltonianCompact:
    """Same interface as Hamiltonian, but H2 is kept in the compact layout
    Lc[(K2,b2), b1, g] (nb*nk, nb, ng_total); the dense (nb*nk, nb*nk, ng)
    tensor is never built. Each momentum sector q has its own block structure
    K1 = kmap[q, K2], which is used to gather the trial-dependent quantities.

    The trial must be the block-diagonal HF determinant (first ne orbitals of
    every k-point).
    """
    one_body: object        # (nb*nk, nb*nk)
    two_body: object        # (nb*nk, nb, ng_total), raw (unscaled) L
    kmap: object            # (nk, nk), kmap[q, K2] = K1
    sizes: object           # (nk,), columns of every q sector
    H_zero: complex = 0.0
    q_list: object = None   # unused, kept for interface symmetry
    test_random_field: object = None
    exchange_mode: str = "gram"     # 'gram' (per-sector Gram of the occupied blocks) or 'direct' (sum over g, reference)

    # ------------------------------------------------------------------
    def setup_energy_expressions(self, config, trial_det):
        be = config.backend
        self._be = be
        nk, nb, ne = config.num_kpoint, config.num_orbital, config.num_electron
        nbk, nek = nb * nk, ne * nk
        kmap = np.asarray(self.kmap, dtype=np.int64)
        sizes = np.asarray(self.sizes, dtype=np.int64)
        off = sector_offsets(sizes)
        ng = int(sizes.sum())
        if ng != config.num_g:
            raise ValueError(f"sector sizes sum to {ng}, NGVEC={config.num_g}")
        Lc = np.asarray(be.to_numpy(self.two_body), dtype=np.complex128)
        if Lc.shape != (nbk, nb, ng):
            raise ValueError(f"compact H2 has shape {Lc.shape}, expected {(nbk, nb, ng)}")
        h1_host = be.to_numpy(self.one_body).astype(np.complex128)
        trial_host = be.to_numpy(trial_det)
        if not np.allclose(trial_host, block_diag(*([np.eye(nb, ne)] * nk))):
            raise NotImplementedError("compact H2 requires the block-diagonal HF trial")
        shape_theta = (config.num_walkers, nbk, nek)

        # ---- one-body energy (as in the dense class) -------------------
        h1_trial = contract("pi, pq -> iq", trial_det, self.one_body)
        self._one_body_expression = contract_expression(
            "ip, wpi -> w", h1_trial, shape_theta, constants=[0], optimize="greedy")

        # ---- exchange: Gram over g of the occupied blocks, one matrix per sector q ------------------------------
        # G[q, K2, Kb, (b2,ob), (oa,bp)] = sum_{g in q} Lc[(K2,b2),oa,g] conj(Lc[(Kb,ob),bp,g]); the row k-points of
        # both factors (kmap[q,K2], kmap[q,Kb]) are implicit, so no zero blocks are stored.
        occ_rows = np.array([K * nb + o for K in range(nk) for o in range(ne)])
        G = np.zeros((nk, nk, nk, nb * ne, ne * nb), dtype=np.complex128)
        for q in range(nk):
            cols = slice(off[q], off[q + 1])
            A = Lc[:, :ne, cols].reshape(nbk * ne, -1)                 # (K2,b2,oa)
            B = Lc[occ_rows][:, :, cols].reshape(nek * nb, -1)         # (Kb,ob,bp)
            Gq = (A @ B.conj().T).reshape(nk, nb, ne, nk, ne, nb)      # K2,b2,oa,Kb,ob,bp
            G[q] = Gq.transpose(0, 3, 1, 4, 2, 5).reshape(nk, nk, nb * ne, ne * nb)
        self._G = be.array(G, dtype=config.complex_type)
        del G
        self._singularity_correction = config.singularity * config.num_kpoint
        self._kq = [np.asarray(kmap[q]) for q in range(nk)]
        self._kinv = [np.argsort(kmap[q]) for q in range(nk)]            # K2 with kmap[q,K2] = K1

        # ---- mean field ------------------------------------------------
        fixed = kmap == np.arange(nk)[None, :]            # fixed[q, K]: block (K, K) lives in sector q
        mf_sectors = [q for q in range(nk) if fixed[q].any()]
        for q in mf_sectors:
            if not fixed[q].all():
                raise NotImplementedError(
                    "sector with only some diagonal blocks: not a regular k-mesh")
        L0 = np.zeros(ng, dtype=np.complex128)
        for q in mf_sectors:
            cols = slice(off[q], off[q + 1])
            for K in range(nk):
                L0[cols] += np.einsum("iig->g", Lc[K * nb:K * nb + ne, :ne, cols])
        self.H_zero = 2 * np.einsum("g,g->", L0, L0.conj()) / (2 * ne * nk**2)

        change = np.zeros((nbk, nbk), dtype=np.complex128)
        for q in mf_sectors:
            cols = slice(off[q], off[q + 1])
            avg = 2 * L0[cols]
            for K in range(nk):
                blk = Lc[K * nb:(K + 1) * nb, :, cols]                # blk[p, r, g] = L[(K,r),(K,p),g]
                change[K * nb:(K + 1) * nb, K * nb:(K + 1) * nb] += \
                    np.einsum("prg,g->rp", blk, avg.conj()) + np.einsum("rpg,g->rp", blk.conj(), avg)
        h_sic = np.zeros((nbk, nbk), dtype=np.complex128)
        for q in range(nk):
            cols = slice(off[q], off[q + 1])
            for K2 in range(nk):
                K = kmap[q, K2]
                A = Lc[K2 * nb:(K2 + 1) * nb, :, cols]                 # (b2, b1, g)
                h_sic[K * nb:(K + 1) * nb, K * nb:(K + 1) * nb] += np.einsum("cbg,cdg->bd", A, A.conj())
        h_sic *= -1.0 / (2 * nk)
        h1_total = h1_host + change / (2 * nk) + h_sic
        self._L0, self._change, self._h_sic = L0, change, h_sic     # small (ng, nbk^2) pieces, kept for diagnostics

        h1_total_be = be.array(h1_total, dtype=config.complex_type)
        self._h1 = -h1_total_be * config.timestep
        self._exp_h1 = be.expm(-h1_total_be * config.timestep)
        self._exp_h1_half = be.expm(-0.5 * h1_total_be * config.timestep)
        self._sqrt_tau = be.sqrt(be.array(config.timestep)).astype(config.float_type)

        # ---- sector bookkeeping (columns are padded to the largest sector) ----
        nmax = int(sizes.max())
        self._nk, self._nb, self._ne, self._ng, self._nmax = nk, nb, ne, ng, nmax

        ar_nk = np.arange(nk)
        qmap = np.zeros((nk, nk), dtype=np.int64)                                               # qmap[K1,K2] = q
        for q in range(nk):
            for K2 in range(nk):
                qmap[kmap[q, K2], K2] = q
        self._qmap, self._qmapT = qmap, qmap.T.copy()
        self._ar = ar_nk
        gidx = np.full((nk, nmax), ng, dtype=np.int64)                  # padded -> index of the zero row
        valid = np.zeros(ng, dtype=np.int64)
        for q in range(nk):
            gidx[q, :sizes[q]] = np.arange(off[q], off[q + 1])
            valid[off[q]:off[q + 1]] = q * nmax + np.arange(sizes[q])
        self._gidx, self._valid = gidx, valid

        # ---- single padded compact tensor for batched matrix products -------------------------------------
        # M[q, g, (K2,b2,b1)] = Lc[(K2,b2), b1, off_q + g]: rows are the columns g of sector q.
        M = np.zeros((nk, nmax, nk * nb * nb), dtype=np.complex128)
        for q in range(nk):
            n = sizes[q]
            M[q, :n, :] = Lc[:, :, off[q]:off[q + 1]].reshape(nk * nb * nb, n).T
        self._M = be.array(M, dtype=config.complex_type)
        del M
        # mean-field shift: L_mf = L - c_g on the diagonal blocks of the sector(s) with fixed points
        c = np.zeros(ng, dtype=np.complex128)
        for q in mf_sectors:
            c[off[q]:off[q + 1]] = L0[off[q]:off[q + 1]] / (ne * nk)
        self._c = be.array(c[:, None], dtype=config.complex_type)
        self._c_conj = be.array(c.conj()[:, None], dtype=config.complex_type)
        self._has_mf = bool(np.any(c != 0))
        # theta gathers (theta^T has shape (nbk, nek, w)); masks zero the unused occupied slot
        k_ = np.arange(nk)[None, :, None, None]
        q_ = np.arange(nk)[:, None, None, None]
        b2_ = np.arange(nb)[None, None, :, None]
        b1_ = np.arange(nb)[None, None, None, :]
        km = kmap[:, :, None, None]
        self._rP = np.broadcast_to(k_ * nb + b2_, (nk, nk, nb, nb)).copy()                 # row (K2, b2)
        self._cP = np.broadcast_to(km * ne + np.minimum(b1_, ne - 1), (nk, nk, nb, nb)).copy()   # col (kmap[q,K2], o=b1)
        self._mP = be.array((np.arange(nb) < ne).astype(float)[None, None, None, :, None], dtype=config.float_type)
        self._rQ = np.broadcast_to(km * nb + b1_, (nk, nk, nb, nb)).copy()                 # row (kmap[q,Ki], b1)
        self._cQ = np.broadcast_to(k_ * ne + np.minimum(b2_, ne - 1), (nk, nk, nb, nb)).copy()   # col (Ki, o=b2)
        self._mQ = be.array((np.arange(nb) < ne).astype(float)[None, None, :, None, None], dtype=config.float_type)
        self._occ_rows = np.array([K * nb + o for K in range(nk) for o in range(ne)])
        self._ar_nek = np.arange(nek)
        self._eye = be.eye(nbk, dtype=config.complex_type)
        self._inv_sqrt_nk = float(1.0 / np.sqrt(nk))      # python float: keeps complex64 as complex64
        self.two_body = None          # the raw tensor is not needed any more: only M, G and small constants stay

    # ------------------------------------------------------------------
    def compute_one_body(self, theta):
        return 2 * self._one_body_expression(theta)

    def compute_hartree(self, theta):
        """E_H = 2 sum_g P_g Q_g, with the same two contractions as the force bias (raw L, no extra tensor)."""
        P, Q = self._pq(theta)
        return 2 * (P * Q).sum(axis=0)

    def compute_exchange(self, theta):
        e = self._exchange_direct(theta) if self.exchange_mode == "direct" else self._exchange_gram(theta)
        return -e + self._singularity_correction

    def _exchange_gram(self, theta):
        """sum_q sum_{K2,Kb} theta[(K2,b2),(Kb,ob)] theta[(kmap[q,Kb],bp),(kmap[q,K2],oa)] G_q[(K2,b2,oa),(Kb,ob,bp)]."""
        nk, nb, ne = self._nk, self._nb, self._ne
        w = theta.shape[0]
        tv = theta.reshape(w, nk, nb, nk, ne)                                      # (w,K,b,K',o)
        th1 = tv.transpose(1, 3, 0, 2, 4).reshape(nk * nk, w, nb * ne)             # (K2,Kb | w | b2,ob)
        total = 0
        for q in range(nk):
            kq = self._kq[q]
            th2 = tv[:, kq][:, :, :, kq]                                           # (w, Kb, bp, K2, oa)
            th2 = th2.transpose(3, 1, 0, 4, 2).reshape(nk * nk, w, ne * nb)        # (K2,Kb | w | oa,bp)
            X = self._be.matmul(th1, self._G[q].reshape(nk * nk, nb * ne, ne * nb))  # (K2,Kb | w | oa,bp)
            total = total + (X * th2).sum(axis=(0, 2))
        return total

    def _exchange_direct(self, theta, max_elements=2 * 10**8):
        """Reference form: sum over every column g, E = sum_g sum_{ij} Y_g[j,i] Z_g[i,j] (no Gram precontraction).
        Y_g[(Kj,oj),(Ki,oi)] = sum_b2 L[(Kj,oj),(K2,b2),g] theta[(K2,b2),(Ki,oi)], K2 = kmap^-1[q,Kj],
        Z_g[(Ki,oi),(Kj,oj)] = sum_bp conj(L[(Ki,bp'),(Ki,oi),g]) theta[(kmap[q,Ki],bp),(Kj,oj)]. Walkers are chunked."""
        nk, nb, ne, nmax = self._nk, self._nb, self._ne, self._nmax
        be = self._be
        w = theta.shape[0]
        tv = theta.reshape(w, nk, nb, nk, ne)
        chunk = max(1, int(max_elements // (nmax * (nk * ne) ** 2)))
        out = []
        for w0 in range(0, w, chunk):
            tvc = tv[w0:w0 + chunk]
            total = 0
            for q in range(nk):
                A = self._M[q].reshape(nmax, nk, nb, nb)                               # g, K2, b2, b1
                kq, ki = self._kq[q], self._kinv[q]
                Y = contract("gkbo, wkbiu -> wgkoiu", A[:, ki, :, :ne], tvc[:, ki])    # w,g,Kj,oj,Ki,oi
                Z = contract("gkib, wkbju -> wgkiju", A[:, :, :ne, :].conj(), tvc[:, kq])  # w,g,Ki,oi,Kj,oj
                total = total + contract("wgabcd, wgcdab -> w", Y, Z)
            out.append(total)
        return be.concatenate(out, axis=0) if len(out) > 1 else out[0]

    def create_random_field(self, config):
        if self.test_random_field is None:
            return config.backend.random_normal(
                (2 * config.num_g, config.num_walkers), config.float_type)
        return self.test_random_field

    # ------------------------------------------------------------------
    def _pq(self, theta):
        """P[g,w] = sum theta[w,r,i] L[i_occ,r,g] and Q[g,w] = sum theta[w,r,i] conj(L[r,i_occ,g]) for the raw L.

        Both are batched matrix products over the sector index q with the single padded tensor M:
        for every (q, K2) only the block K1 = kmap[q, K2] of theta is touched."""
        nk, nb = self._nk, self._nb
        thT = theta.transpose(1, 2, 0)                                    # (nbk, nek, w)
        w = thT.shape[-1]
        gP = thT[self._rP, self._cP] * self._mP                           # q,K2,b2,b1,w : theta[(K2,b2),(kmap,o=b1)]
        gQ = thT[self._rQ, self._cQ] * self._mQ                           # q,Ki,b2,b1,w : theta[(kmap,b1),(Ki,o=b2)]
        rhs = self._be.concatenate([gP.reshape(nk, nk * nb * nb, w), gQ.conj().reshape(nk, nk * nb * nb, w)], axis=-1)
        out = self._be.matmul(self._M, rhs)                               # q, g, 2w
        P = out[..., :w].reshape(nk * self._nmax, w)[self._valid]
        Q = out[..., w:].conj().reshape(nk * self._nmax, w)[self._valid]
        return P, Q

    def _force_bias(self, theta):
        """(2*ng, nw): [ (P+Q)/2 ; i(P-Q)/2 ] / sqrt(nk) for the mean-field subtracted L."""
        P, Q = self._pq(theta)
        if self._has_mf:
            S = theta[:, self._occ_rows, self._ar_nek].sum(axis=1)       # sum_i theta[(i_occ), i] (= nek for the HF trial)
            P = P - self._c * S[None, :]
            Q = Q - self._c_conj * S[None, :]
        s = self._inv_sqrt_nk
        return self._be.concatenate([(P + Q) * (s / 2), (P - Q) * (1j * s / 2)], axis=0)

    def _auxiliary_field(self, x):
        """sum_g A_e x_e + A_o x_o as one dense (nb*nk, nb*nk, nw) matrix, A_e = (L+L^+)/2, A_o = i(L-L^+)/2,
        built block by block: sector q and column k-point K2 fill the block (kmap[q,K2], K2); no dense L_g is formed."""
        nk, nb, ng = self._nk, self._nb, self._ng
        nw = x.shape[1]
        be = self._be
        u = (x[:ng] + 1j * x[ng:]) / 2
        v = (x[:ng] - 1j * x[ng:]) / 2
        zero = be.zeros((1, nw), dtype=u.dtype)
        u_pad = be.concatenate([u, zero], axis=0)[self._gidx]                     # q,g,w
        v_pad = be.concatenate([v, zero], axis=0)[self._gidx]
        rhs = be.concatenate([u_pad, v_pad.conj()], axis=-1)                       # q,g,2w
        out = be.matmul(self._M.transpose(0, 2, 1), rhs)                           # q,(K2,b2,b1),2w
        R1 = out[..., :nw].reshape(nk, nk, nb, nb, nw)                             # sum_g L u         [q,K2,b2,b1,w]
        R2 = out[..., nw:].conj().reshape(nk, nk, nb, nb, nw)                      # sum_g conj(L) v
        F1 = R1[self._qmap, self._ar[None, :]].transpose(0, 3, 1, 2, 4)            # K1,b1,K2,b2,w
        F2 = R2[self._qmapT, self._ar[:, None]].transpose(0, 2, 1, 3, 4)           # Ki,bi,Kj,bj,w
        field = (F1 + F2).reshape(nk * nb, nk * nb, nw)
        if self._has_mf:       # L_mf = L - c_g on the diagonal: subtract (sum_g c_g u_g + conj(c_g) v_g) * identity
            sw = (self._c * u).sum(axis=0) + (self._c_conj * v).sum(axis=0)
            field = field - self._eye[:, :, None] * sw[None, None, :]
        return field * self._inv_sqrt_nk

    def create_auxiliary_field(self, config, theta):
        random_field = self.create_random_field(config)
        force_bias = -2j * self._sqrt_tau * self._force_bias(theta)
        force_bias = config.backend.where(abs(force_bias) > 1, 0.0, force_bias)
        arg = contract("gw, gw -> w", random_field - 0.5 * force_bias, force_bias)
        field = 1j * self._sqrt_tau * self._auxiliary_field(random_field - force_bias)
        return field, config.backend.exp(arg)

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
    h1_per_k = np.load(os.path.expanduser(filename)).astype(np.complex128)
    # Expected shape (num_orb, num_orb, num_k) — what opt/vasp produce.
    if h1_per_k.ndim == 2:
        h1_per_k = h1_per_k[:, :, None]
    h1 = reshape_H1(h1_per_k, config.num_kpoint, config.num_orbital)
    return config.backend.array(h1, dtype=config.complex_type)


def obtain_H2(config, filename="H2_zip.npy"):
    """Load H2 in shape (num_orb*num_k, num_orb*num_k, num_g_total)."""
    h2 = np.load(os.path.expanduser(filename)).astype(np.complex128)
    return config.backend.array(h2, dtype=config.complex_type)


def obtain_Q_list(config, filename="Q_list.npy"):
    """Load Q-list. Layout: rows are [K1, K2, Q]. Falls back to heuristic."""
    filename = os.path.expanduser(filename)
    if not os.path.exists(filename):
        return build_default_q_list(config.num_kpoint)
    ql = np.load(filename)
    # Files coming from the legacy pipeline are (3, N) — transpose if so.
    if ql.shape[0] == 3 and ql.shape[1] != 3:
        ql = ql.T
    return ql.astype(np.int64, copy=False)


# =========================================================================
#  Compact H2 layout
#
#  Full:     L[(K1,b1), (K2,b2), g]     shape (nb*nk, nb*nk, ng_total)
#  Compact:  Lc[(K2,b2), b1, g]         shape (nb*nk, nb,    ng_total)
#
#  The columns g are grouped in sectors of momentum transfer q (sector q
#  holds the n_q columns [off_q, off_q + n_q)). For a given column k-point
#  K2 and sector q the row k-point is fixed by momentum conservation,
#  K1 = kmap[q, K2], and all other (K1, K2) blocks vanish, so
#
#       Lc[(K2,b2), b1, g] = L[(kmap[q(g),K2], b1), (K2,b2), g].
#
#  This is the layout written by the reduced VASP exporter, (ng*nk, nb,
#  nb*nk) in numpy order, with the axes reversed.
# =========================================================================
def build_kmap(q_list, num_k):
    """kmap[q, K2] = K1 (all zero-based) from the rows [K1, K2, Q] (one-based)
    of the Q-list. Requires a unique K1 for every (K2, Q), i.e. exact
    momentum conservation, and a complete list."""
    kmap = -np.ones((num_k, num_k), dtype=np.int64)
    for K1, K2, Q in np.asarray(q_list, dtype=np.int64):
        old = kmap[Q - 1, K2 - 1]
        if old >= 0 and old != K1 - 1:
            raise ValueError(
                f"Q-list maps (K2={K2}, Q={Q}) to two row k-points "
                f"({old + 1} and {K1}); compact H2 needs momentum conservation."
            )
        kmap[Q - 1, K2 - 1] = K1 - 1
    if np.any(kmap < 0):
        raise ValueError("Q-list is incomplete: not every (K2, Q) has a K1.")
    return kmap


def sector_sizes_default(num_g_total, num_k):
    """Equal split of the columns over the q sectors (uncompressed export)."""
    if num_g_total % num_k != 0:
        raise ValueError(
            f"ng_total={num_g_total} is not divisible by num_k={num_k}; "
            "supply the per-sector column counts (Q_sizes.npy)."
        )
    return np.full(num_k, num_g_total // num_k, dtype=np.int64)


def sector_offsets(sizes):
    return np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)


def is_compact_shape(shape, num_orb, num_k):
    """True for (nb*nk, nb, ng). For num_k == 1 both layouts coincide and the
    array is identical, so it is treated as dense."""
    return len(shape) == 3 and num_k > 1 and shape[0] == num_orb * num_k \
        and shape[1] == num_orb


def dense_to_compact(h2_dense, kmap, sizes, num_orb):
    """(nb*nk, nb*nk, ng) -> (nb*nk, nb, ng). Blocks that violate momentum
    conservation must vanish; this is checked."""
    nk = kmap.shape[0]
    off = sector_offsets(sizes)
    out = np.zeros((num_orb * nk, num_orb, h2_dense.shape[2]), dtype=h2_dense.dtype)
    kept = 0.0
    for q in range(nk):
        cols = slice(off[q], off[q + 1])
        for K2 in range(nk):
            K1 = kmap[q, K2]
            blk = h2_dense[K1 * num_orb:(K1 + 1) * num_orb,
                           K2 * num_orb:(K2 + 1) * num_orb, cols]
            out[K2 * num_orb:(K2 + 1) * num_orb, :, cols] = blk.transpose(1, 0, 2)
            kept += np.sum(np.abs(blk) ** 2)
    total = np.sum(np.abs(h2_dense) ** 2)
    if abs(total - kept) > 1e-9 * max(total, 1.0):
        raise ValueError("dense H2 has weight outside the momentum-conserving blocks")
    return out


def compact_to_dense(h2_compact, kmap, sizes, num_orb):
    """(nb*nk, nb, ng) -> (nb*nk, nb*nk, ng)."""
    nk = kmap.shape[0]
    off = sector_offsets(sizes)
    out = np.zeros((num_orb * nk, num_orb * nk, h2_compact.shape[2]),
                   dtype=h2_compact.dtype)
    for q in range(nk):
        cols = slice(off[q], off[q + 1])
        for K2 in range(nk):
            K1 = kmap[q, K2]
            out[K1 * num_orb:(K1 + 1) * num_orb,
                K2 * num_orb:(K2 + 1) * num_orb, cols] = \
                h2_compact[K2 * num_orb:(K2 + 1) * num_orb, :, cols].transpose(1, 0, 2)
    return out


def obtain_Q_sizes(config, filename="Q_sizes.npy"):
    """Number of retained columns per q sector. Falls back to an equal split."""
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        sizes = np.load(filename).astype(np.int64).ravel()
        if len(sizes) != config.num_kpoint or sizes.sum() != config.num_g:
            raise ValueError(
                f"{filename}: sizes {sizes.tolist()} inconsistent with "
                f"num_k={config.num_kpoint}, NGVEC={config.num_g}")
        return sizes
    return sector_sizes_default(config.num_g, config.num_kpoint)


def obtain_H2_host(config, filename="H2_zip.npy"):
    """Load H2 (dense or compact) as a complex128 numpy array and report the
    layout: returns (array, 'dense' | 'compact')."""
    h2 = np.load(os.path.expanduser(filename)).astype(np.complex128)
    if h2.shape[2] != config.num_g:
        raise ValueError(f"H2 has {h2.shape[2]} columns, NGVEC={config.num_g}")
    layout = "compact" if is_compact_shape(
        h2.shape, config.num_orbital, config.num_kpoint) else "dense"
    return h2, layout


def build_hamiltonian(config, one_body, h2_host, layout, q_list, q_sizes=None):
    """Hamiltonian for a dense or compact H2 (layout from obtain_H2_host).

    dense   -> Hamiltonian (reference path, dense (nb*nk, nb*nk, ng) tensor)
    compact -> HamiltonianCompact (momentum-conserving blocks only)
    """
    if layout == "compact":
        kmap = build_kmap(q_list, config.num_kpoint)
        if q_sizes is None:
            q_sizes = sector_sizes_default(config.num_g, config.num_kpoint)
        # the compact tensor stays on the host in double precision until setup has built the working tensors
        return HamiltonianCompact(one_body=one_body, two_body=h2_host, kmap=kmap, sizes=q_sizes, q_list=q_list)
    return Hamiltonian(
        one_body=one_body,
        two_body=config.backend.array(h2_host, dtype=config.complex_type),
        q_list=q_list)


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
    energy = e1 + (eh + ex) / config.num_kpoint
    weighted_energy = energy @ walkers.weights
    sum_weights = config.backend.sum(walkers.weights)
    weighted_energy_global = config.comm.allreduce(weighted_energy)
    sum_weights_global = config.comm.allreduce(sum_weights)
    return (
        weighted_energy_global / sum_weights_global,
        weighted_energy_global,
        sum_weights_global,
    )


def measure_hartree(config, trial, walkers, hamiltonian):
    """Hartree energy averaged over walkers; used to compute h_0 for S2 propagator."""
    th = biorthogonalize(config.backend, trial, walkers.slater_det)
    energy_hartree = hamiltonian.compute_hartree(th)
    energy = energy_hartree / config.num_kpoint
    weighted_energy = energy @ walkers.weights
    sum_weights = config.backend.sum(walkers.weights)
    weighted_energy_global = config.comm.allreduce(weighted_energy)
    sum_weights_global = config.comm.allreduce(sum_weights)
    return weighted_energy_global / sum_weights_global


def measure_components(config, trial, walkers, hamiltonian):
    """Returns (E_one, Hartree, Exchange) — useful for verification."""
    th = biorthogonalize(config.backend, trial, walkers.slater_det)
    e1 = hamiltonian.compute_one_body(th)
    eh = hamiltonian.compute_hartree(th)
    ex = hamiltonian.compute_exchange(th)
    w = walkers.weights
    sumw = config.backend.sum(w)
    return (
        (e1 @ w) / sumw,
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

    h_0 = exp(dtau * H_zero) — scalar applied to the new walker determinant.
    e_0 is the running energy estimate used in the importance reweighting.
    """
    new_walkers = Walkers(
        config.backend.zeros_like(walkers.slater_det),
        config.backend.zeros_like(walkers.weights),
    )
    th = biorthogonalize(config.backend, trial, walkers.slater_det)
    h2, importance = hamiltonian.create_auxiliary_field(config, th)
    num_rare_event = 0

    if config.propagator == "Taylor":
        h = hamiltonian.h1 + h2
        full = apply_taylor(config, h, walkers.slater_det)
        new_walkers.slater_det = h_0 * full
    elif config.propagator == "S1":
        full_h2 = apply_taylor(config, h2, walkers.slater_det)
        full_h1 = hamiltonian.exp_h1 @ full_h2
        new_walkers.slater_det = h_0 * full_h1
    elif config.propagator == "S2":
        half = hamiltonian.exp_h1_half @ walkers.slater_det
        full_h2 = apply_taylor(config, h2, half)
        half = hamiltonian.exp_h1_half @ full_h2
        new_walkers.slater_det = h_0 * half
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
def cholesky_orthonormalize_complex_jax(walkers):
    """Orthonormalize walker columns via Cholesky decomposition (JAX / GPU path).

    walkers: [batch, m, n] complex JAX array.
    Computes Q such that Q^H Q = I using A^H A = L L^H, then Q = A (L^H)^{-1}.
    Faster than QR on Ampere/Ada GPUs for large n.
    """
    import jax
    import jax.numpy as jnp

    def ortho_single(A):
        AhA = A.conj().T @ A                                    # [n, n]
        L = jnp.linalg.cholesky(AhA)                           # [n, n]
        Lh_inv = jnp.linalg.solve(L.conj().T,
                                  jnp.eye(L.shape[-1], dtype=A.dtype))
        return A @ Lh_inv                                       # [m, n]

    return jax.vmap(ortho_single)(walkers)


def reortho_qr(config, walker_matrix):
    """Reorthogonalize walkers. Uses Cholesky on JAX (GPU-optimised), QR otherwise."""
    if isinstance(config.backend, JaxBackend):
        return cholesky_orthonormalize_complex_jax(walker_matrix)
    Q, _ = config.backend.qr(walker_matrix)
    return Q


def init_walkers_weights(config, n_walkers):
    return config.backend.ones(n_walkers, dtype=config.complex_type)


def _rebalance_comb_jax(config, weights):
    """Systematic resampling kept entirely on the JAX device (no CPU transfer)."""
    import jax.numpy as jnp
    c = config.backend.cumsum(weights.real)
    N = len(c)
    W = c[-1]
    r = config.backend.random_uniform((), dtype=jnp.float32) * (W / N)
    U = r + jnp.arange(N) * (W / N)
    return jnp.searchsorted(c, U, side="left")


def rebalance_comb(config, weights):
    """Systematic resampling on a single rank; returns new walker indices.

    Uses a GPU-resident path for JAX (avoids device→host transfer) and a
    NumPy/CuPy path otherwise.
    """
    if isinstance(config.backend, JaxBackend):
        return _rebalance_comb_jax(config, weights)
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
    # LOCAL FIX: buffers below are complex128; MPI Gather with complex64 (Single precision) input
    # reinterprets bytes -> garbage/inf weights. Work in complex128 here (caller casts back).
    walkers_mats_up = np.asarray(walkers_mats_up, dtype=np.complex128)
    walkers_weights = np.asarray(walkers_weights, dtype=np.complex128)
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
