"""
vafpy_v3: unified GPU/CPU AFQMC with k-point support.

Backend abstraction (NumPy / JAX / CuPy) follows the vafpy_v1 design.
K-point physics (mean-field subtraction, Q-list, block-diagonal trial)
follows the legacy reference in Code/opt.
"""
from dataclasses import dataclass
import os
import secrets
import types
from time import time

import numpy as np
import scipy
from scipy.linalg import expm, block_diag
from mpi4py import MPI
from opt_einsum import contract, contract_expression

from timing import NULL_TIMERS


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
    h1 = np.zeros([num_orb * num_k, num_orb * num_k], dtype=H1.dtype)
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
    result = np.zeros(h2.shape[2], dtype=h2.dtype)
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
    change = np.zeros_like(h1)
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
def run_seed(value, comm):
    """Seed of the run from SEED in vafpy.in: an integer, or 'random' (also when SEED is missing), in which case rank 0
    draws 63 bits from the operating system and broadcasts them, so that all ranks share one seed that is printed.
    Returns (seed, source) with source 'input' or 'random'."""
    if value is None or str(value).strip().lower() == "random":
        seed = secrets.randbits(63) if comm.Get_rank() == 0 else None
        return comm.bcast(seed, root=0), "random"
    return int(value), "input"


def seed_sequence(seed, rank=None):
    """SeedSequence of one stream: for an MPI rank the child `rank` of SeedSequence(seed), i.e. the same as
    SeedSequence(seed).spawn(n_ranks)[rank]; independent streams for all ranks, and different seeds never overlap."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed) if rank is None else np.random.SeedSequence(seed, spawn_key=(rank,))


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
        self._rng = np.random.default_rng(seed_sequence(seed))   # PCG64

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
        # threefry key from two 32-bit words of the SeedSequence (all 63 bits of the seed and the rank enter)
        self._key = jax.random.wrap_key_data(np.asarray(seed_sequence(seed).generate_state(2, np.uint32)))

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
        self._rng = cp.random.default_rng(int(seed_sequence(seed).generate_state(1, np.uint64)[0]))

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
    """seed: an integer or a SeedSequence (seed_sequence(run seed, rank) for the stream of one MPI rank)."""
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
    spin_polarized: bool = False      # False: closed shell, one determinant for both spins
    num_electron_dn: int = None       # electrons per k-point with spin down (spin_polarized only; default num_electron)

    @property
    def channels(self):
        """Spin channels as (electrons per k-point, multiplicity). Closed shell: one channel counted twice.
        Spin-unrestricted: spin up and (if it has electrons) spin down, each counted once."""
        if not self.spin_polarized:
            return [(self.num_electron, 2)]
        ne_dn = self.num_electron if self.num_electron_dn is None else self.num_electron_dn
        return [(self.num_electron, 1)] + ([(ne_dn, 1)] if ne_dn > 0 else [])

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
    slater_det: object                # closed shell: the determinant of both spins; spin-unrestricted: spin up
    weights: object
    slater_det_dn: object = None      # spin-unrestricted only: spin down

    @property
    def dets(self):
        """Determinants of the spin channels, in the order of Configuration.channels."""
        return [self.slater_det] if self.slater_det_dn is None else [self.slater_det, self.slater_det_dn]

    @dets.setter
    def dets(self, dets):
        self.slater_det = dets[0]
        self.slater_det_dn = dets[1] if len(dets) > 1 else None


def as_list(x):
    """A per-channel quantity as a list (a single array is the closed-shell case)."""
    return list(x) if isinstance(x, (list, tuple)) else [x]


def unlist(xs):
    """Inverse of as_list: one channel is passed on as the bare array."""
    return xs[0] if len(xs) == 1 else list(xs)


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

    def create_auxiliary_field(self, config, theta, timers=NULL_TIMERS):
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


def make_compact_kernels(be, nk, nb, ne, nmax, ng, kmap, sizes, has_mf, jit=False):
    """Pure functions (explicit array arguments) for the H2-dependent contractions of the compact layout.

    M[q, g, (K2,b2,b1)]  padded compact tensor,  G[q, K2, Kb, (b2,ob), (oa,bp)]  per-sector Gram of the occupied
    blocks, c = mean-field shift. Only the blocks K1 = kmap[q,K2] of theta are touched, theta is never expanded
    nk-fold, and no (nb*nk, nb*nk, ng) tensor appears. With jit=True (JAX) every function is compiled so that the
    gathers, transposes and products fuse."""
    nbk, nek = nk * nb, nk * ne
    kmap = np.asarray(kmap, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    off = sector_offsets(sizes)
    inv_sqrt = float(1.0 / np.sqrt(nk))                        # python float: keeps complex64 as complex64
    q_, k_ = np.arange(nk)[:, None, None, None], np.arange(nk)[None, :, None, None]
    b_, o_ = np.arange(nb), np.arange(ne)
    km = kmap[:, :, None, None]
    # theta gathers (theta^T has shape (nbk, nek, w)); only blocks (K2, kmap[q,K2]) are addressed
    rP = np.broadcast_to(k_ * nb + b_[None, None, :, None], (nk, nk, nb, ne)).copy()   # row (K2,b2)
    cP = np.broadcast_to(km * ne + o_[None, None, None, :], (nk, nk, nb, ne)).copy()   # col (kmap[q,K2], o)
    rQ = np.broadcast_to(km * nb + b_[None, None, None, :], (nk, nk, ne, nb)).copy()   # row (kmap[q,Ki], b1)
    cQ = np.broadcast_to(k_ * ne + o_[None, None, :, None], (nk, nk, ne, nb)).copy()   # col (Ki, o)
    qmap = np.zeros((nk, nk), dtype=np.int64)                  # qmap[K1,K2] = q
    for q in range(nk):
        for K2 in range(nk):
            qmap[kmap[q, K2], K2] = q
    qmapT, ar = qmap.T.copy(), np.arange(nk)
    gidx = np.full((nk, nmax), ng, dtype=np.int64)             # padded slot -> index of the zero row
    valid = np.zeros(ng, dtype=np.int64)
    for q in range(nk):
        gidx[q, :sizes[q]] = np.arange(off[q], off[q + 1])
        valid[off[q]:off[q + 1]] = q * nmax + np.arange(sizes[q])
    occ_rows, ar_nek = np.array([K * nb + o for K in range(nk) for o in range(ne)]), np.arange(nek)
    kq = [kmap[q] for q in range(nk)]

    def pq(M, theta):
        """P[g,w] = sum theta[w,r,i] L[i_occ,r,g],  Q[g,w] = sum theta[w,r,i] conj(L[r,i_occ,g])  (raw L):
        batched matrix products over q; P uses the slice b1 < ne of M, Q the slice b2 < ne."""
        M5 = M.reshape(nk, nmax, nk, nb, nb)                                # q, g, K2, b2, b1
        MP = M5[:, :, :, :, :ne].reshape(nk, nmax, nk * nb * ne)            # (K2, b2, o)
        MQ = M5[:, :, :, :ne, :].reshape(nk, nmax, nk * ne * nb)            # (Ki, o, b1)
        thT = theta.transpose(1, 2, 0)                                      # (nbk, nek, w)
        w = thT.shape[-1]
        gP = thT[rP, cP].reshape(nk, nk * nb * ne, w)                       # theta[(K2,b2),(kmap[q,K2],o)]
        gQ = thT[rQ, cQ].reshape(nk, nk * ne * nb, w)                       # theta[(kmap[q,Ki],b1),(Ki,o)]
        P = be.matmul(MP, gP).reshape(nk * nmax, w)[valid]
        Q = be.matmul(MQ, gQ.conj()).conj().reshape(nk * nmax, w)[valid]
        return P, Q

    def force_bias(M, c, cc, theta):
        P, Q = pq(M, theta)
        if has_mf:
            S = theta[:, occ_rows, ar_nek].sum(axis=1)                      # sum_i theta[(i_occ), i]
            P = P - c * S[None, :]
            Q = Q - cc * S[None, :]
        return be.concatenate([(P + Q) * (inv_sqrt / 2), (P - Q) * (1j * inv_sqrt / 2)], axis=0)

    def hartree(M, theta):
        P, Q = pq(M, theta)
        return 2 * (P * Q).sum(axis=0)

    def aux(M, c, cc, eye, x):
        nw = x.shape[1]
        u = (x[:ng] + 1j * x[ng:]) / 2
        v = (x[:ng] - 1j * x[ng:]) / 2
        zero = be.zeros((1, nw), dtype=u.dtype)
        u_pad = be.concatenate([u, zero], axis=0)[gidx]                     # q, g, w
        v_pad = be.concatenate([v, zero], axis=0)[gidx]
        out = be.matmul(M.transpose(0, 2, 1), be.concatenate([u_pad, v_pad.conj()], axis=-1))   # q, (K2,b2,b1), 2w
        R1 = out[..., :nw].reshape(nk, nk, nb, nb, nw)                      # sum_g L u         [q,K2,b2,b1,w]
        R2 = out[..., nw:].conj().reshape(nk, nk, nb, nb, nw)               # sum_g conj(L) v
        F1 = R1[qmap, ar[None, :]].transpose(0, 3, 1, 2, 4)                 # K1,b1,K2,b2,w
        F2 = R2[qmapT, ar[:, None]].transpose(0, 2, 1, 3, 4)                # Ki,bi,Kj,bj,w
        field = (F1 + F2).reshape(nbk, nbk, nw)
        if has_mf:                    # L_mf = L - c_g on the diagonal: subtract (sum_g c_g u_g + conj(c_g) v_g) * 1
            sw = (c * u).sum(axis=0) + (cc * v).sum(axis=0)
            field = field - eye[:, :, None] * sw[None, None, :]
        return field * inv_sqrt

    def exchange(G, theta):
        """sum_q sum_{K2,Kb} theta[(K2,b2),(Kb,ob)] theta[(kmap[q,Kb],bp),(kmap[q,K2],oa)] G_q[(K2,b2,oa),(Kb,ob,bp)]."""
        w = theta.shape[0]
        tv = theta.reshape(w, nk, nb, nk, ne)                               # (w, K, b, K', o)
        th1 = tv.transpose(1, 3, 0, 2, 4).reshape(nk * nk, w, nb * ne)      # (K2,Kb | w | b2,ob)
        total = 0
        for q in range(nk):
            th2 = tv[:, kq[q]][:, :, :, kq[q]]                              # (w, Kb, bp, K2, oa)
            th2 = th2.transpose(3, 1, 0, 4, 2).reshape(nk * nk, w, ne * nb)  # (K2,Kb | w | oa,bp)
            X = be.matmul(th1, G[q].reshape(nk * nk, nb * ne, ne * nb))
            total = total + (X * th2).sum(axis=(0, 2))
        return total

    fns = {"pq": pq, "force_bias": force_bias, "hartree": hartree, "aux": aux, "exchange": exchange}
    if jit:
        fns = {k: be._jax.jit(f) for k, f in fns.items()}
    return fns


class _Channel:
    """Spin-channel part of the compact Hamiltonian (closed shell: the one channel stands for both spins)."""

    def __init__(self, ne, mult):
        self.ne, self.mult = ne, mult


@dataclass
class HamiltonianCompact:
    """Same interface as Hamiltonian, but H2 is kept in the compact layout
    Lc[(K2,b2), b1, g] (nb*nk, nb, ng_total); the dense (nb*nk, nb*nk, ng)
    tensor is never built. Each momentum sector q has its own block structure
    K1 = kmap[q, K2], which is used to gather the trial-dependent quantities.

    The trial must be the block-diagonal HF determinant (first ne orbitals of
    every k-point) of every spin channel.

    Spin: closed shell (config.channels = [(ne, 2)]) uses one_body/two_body for both spins. Spin-unrestricted
    (config.spin_polarized) uses one_body/two_body for spin up and one_body_dn/two_body_dn for spin down. The two
    spins share the columns g (the same auxiliary field couples to rho_up + rho_dn), so the two compact tensors
    must come from one joint compression with common Q sectors. Per-channel quantities (theta, the field, exp_h1)
    are lists in the order of config.channels; a single channel is passed as the bare array.
    """
    one_body: object        # (nb*nk, nb*nk)
    two_body: object        # (nb*nk, nb, ng_total), raw (unscaled) L
    kmap: object            # (nk, nk), kmap[q, K2] = K1
    sizes: object           # (nk,), columns of every q sector
    H_zero: complex = 0.0
    q_list: object = None   # unused, kept for interface symmetry
    test_random_field: object = None
    exchange_mode: str = "gram"     # 'gram' (per-sector Gram of the occupied blocks) or 'direct' (sum over g, reference)
    one_body_dn: object = None      # spin-unrestricted only
    two_body_dn: object = None

    # ------------------------------------------------------------------
    def setup_energy_expressions(self, config, trial_det):
        be = config.backend
        self._be = be
        wt = np.dtype(config.complex_type)      # host arrays of the setup are in the working precision
        nk, nb = config.num_kpoint, config.num_orbital
        nbk = nb * nk
        kmap = np.asarray(self.kmap, dtype=np.int64)
        sizes = np.asarray(self.sizes, dtype=np.int64)
        off = sector_offsets(sizes)
        ng = int(sizes.sum())
        if ng != config.num_g:
            raise ValueError(f"sector sizes sum to {ng}, NGVEC={config.num_g}")
        trials = as_list(trial_det)
        chans = [_Channel(ne, m) for ne, m in config.channels]
        if len(trials) != len(chans):
            raise ValueError(f"{len(trials)} trial determinants for {len(chans)} spin channels")
        raw = [(self.one_body, self.two_body), (self.one_body_dn, self.two_body_dn)][:len(chans)]
        if len(chans) == 2 and (self.one_body_dn is None or self.two_body_dn is None):
            raise ValueError("spin-unrestricted run without a spin-down Hamiltonian")

        fixed = kmap == np.arange(nk)[None, :]            # fixed[q, K]: block (K, K) lives in sector q
        mf_sectors = [q for q in range(nk) if fixed[q].any()]
        for q in mf_sectors:
            if not fixed[q].all():
                raise NotImplementedError(
                    "sector with only some diagonal blocks: not a regular k-mesh")
        nmax = int(sizes.max())

        for ch, trial, (h1_raw, h2_raw) in zip(chans, trials, raw):
            ne = ch.ne
            nek = ne * nk
            Lc = np.asarray(be.to_numpy(h2_raw), dtype=wt)
            if Lc.shape != (nbk, nb, ng):
                raise ValueError(f"compact H2 has shape {Lc.shape}, expected {(nbk, nb, ng)}")
            ch.h1_host = be.to_numpy(h1_raw).astype(wt)
            trial_host = be.to_numpy(trial)
            if not np.allclose(trial_host, block_diag(*([np.eye(nb, ne)] * nk))):
                raise NotImplementedError("compact H2 requires the block-diagonal HF trial")
            shape_theta = (config.num_walkers, nbk, nek)

            # ---- one-body energy (as in the dense class) -------------------
            h1_trial = contract("pi, pq -> iq", trial, h1_raw)
            ch.one_body_expression = contract_expression(
                "ip, wpi -> w", h1_trial, shape_theta, constants=[0], optimize="greedy")

            # ---- exchange: Gram over g of the occupied blocks, one matrix per sector q --------------------------
            # G[q, K2, Kb, (b2,ob), (oa,bp)] = sum_{g in q} Lc[(K2,b2),oa,g] conj(Lc[(Kb,ob),bp,g]); the row k-points
            # of both factors (kmap[q,K2], kmap[q,Kb]) are implicit, so no zero blocks are stored.
            occ_rows = np.array([K * nb + o for K in range(nk) for o in range(ne)])
            G = np.zeros((nk, nk, nk, nb * ne, ne * nb), dtype=wt)
            for q in range(nk):
                cols = slice(off[q], off[q + 1])
                A = Lc[:, :ne, cols].reshape(nbk * ne, -1)                 # (K2,b2,oa)
                B = Lc[occ_rows][:, :, cols].reshape(nek * nb, -1)         # (Kb,ob,bp)
                Gq = (A @ B.conj().T).reshape(nk, nb, ne, nk, ne, nb)      # K2,b2,oa,Kb,ob,bp
                G[q] = Gq.transpose(0, 3, 1, 4, 2, 5).reshape(nk, nk, nb * ne, ne * nb)
            ch.G = be.array(G, dtype=config.complex_type)
            del G

            # ---- occupied diagonal of the mean-field sector(s) and self-interaction of this spin ------------
            L0 = np.zeros(ng, dtype=wt)
            for q in mf_sectors:
                cols = slice(off[q], off[q + 1])
                for K in range(nk):
                    L0[cols] += np.einsum("iig->g", Lc[K * nb:K * nb + ne, :ne, cols])
            ch.L0 = L0
            h_sic = np.zeros((nbk, nbk), dtype=wt)
            for q in range(nk):
                cols = slice(off[q], off[q + 1])
                for K2 in range(nk):
                    K = kmap[q, K2]
                    A = Lc[K2 * nb:(K2 + 1) * nb, :, cols]                 # (b2, b1, g)
                    h_sic[K * nb:(K + 1) * nb, K * nb:(K + 1) * nb] += np.einsum("cbg,cdg->bd", A, A.conj())
            ch.h_sic = h_sic * (-1.0 / (2 * nk))

            # ---- single padded compact tensor for batched matrix products -------------------------------------
            # M[q, g, (K2,b2,b1)] = Lc[(K2,b2), b1, off_q + g]: rows are the columns g of sector q.
            M = np.zeros((nk, nmax, nk * nb * nb), dtype=wt)
            for q in range(nk):
                n = sizes[q]
                M[q, :n, :] = Lc[:, :, off[q]:off[q + 1]].reshape(nk * nb * nb, n).T
            ch.Lc = Lc
            ch.M = be.array(M, dtype=config.complex_type)
            del M

        # ---- mean field, shared by the spins ---------------------------------------------------------------
        # <rho_g> of the trial: avg = sum over channels of mult * L0 (closed shell: 2 L0); N electrons in total.
        avg = sum(ch.mult * ch.L0 for ch in chans)
        n_tot = sum(ch.mult * ch.ne for ch in chans) * nk
        self.H_zero = np.einsum("g,g->", avg, avg.conj()) / (2 * nk * n_tot)
        # L_mf = L - c_g on the diagonal blocks of the sector(s) with fixed points, c = <rho_g> / N: the same shift
        # for both spins, because the number operator N = N_up + N_dn is what is subtracted.
        c = np.zeros(ng, dtype=wt)
        for q in mf_sectors:
            c[off[q]:off[q + 1]] = avg[off[q]:off[q + 1]] / n_tot
        self._c = be.array(c[:, None], dtype=config.complex_type)
        self._c_conj = be.array(c.conj()[:, None], dtype=config.complex_type)
        self._has_mf = bool(np.any(c != 0))
        self._sqrt_tau = be.sqrt(be.array(config.timestep)).astype(config.float_type)
        for ch in chans:
            change = np.zeros((nbk, nbk), dtype=wt)
            for q in mf_sectors:
                cols = slice(off[q], off[q + 1])
                for K in range(nk):
                    blk = ch.Lc[K * nb:(K + 1) * nb, :, cols]              # blk[p, r, g] = L[(K,r),(K,p),g]
                    change[K * nb:(K + 1) * nb, K * nb:(K + 1) * nb] += \
                        np.einsum("prg,g->rp", blk, avg[cols].conj()) + np.einsum("rpg,g->rp", blk.conj(), avg[cols])
            h1_total = ch.h1_host + change / (2 * nk) + ch.h_sic
            ch.change = change
            h1_total_be = be.array(h1_total, dtype=config.complex_type)
            ch.h1 = -h1_total_be * config.timestep
            ch.exp_h1 = be.expm(-h1_total_be * config.timestep)
            ch.exp_h1_half = be.expm(-0.5 * h1_total_be * config.timestep)
            ch.k = make_compact_kernels(be, nk, nb, ch.ne, nmax, ng, kmap, sizes, self._has_mf,
                                        jit=hasattr(be, "_jax"))
            del ch.Lc, ch.h1_host

        self._chans = chans
        self._singularity_correction = config.singularity * config.num_kpoint
        self._kq = [np.asarray(kmap[q]) for q in range(nk)]
        self._kinv = [np.argsort(kmap[q]) for q in range(nk)]            # K2 with kmap[q,K2] = K1
        self._nk, self._nb, self._ng, self._nmax = nk, nb, ng, nmax
        self._ne = chans[0].ne
        self._eye = be.eye(nbk, dtype=config.complex_type)
        self._inv_sqrt_nk = float(1.0 / np.sqrt(nk))      # python float: keeps complex64 as complex64
        # small pieces kept for diagnostics (first channel; closed shell: the only one)
        self._L0, self._change, self._h_sic = chans[0].L0, chans[0].change, chans[0].h_sic
        self._M, self._G, self._k = chans[0].M, chans[0].G, chans[0].k
        self._h1, self._exp_h1, self._exp_h1_half = chans[0].h1, chans[0].exp_h1, chans[0].exp_h1_half
        # the raw tensors are not needed any more: only M, G and small constants stay
        self.two_body = None
        self.two_body_dn = None

    # ------------------------------------------------------------------
    def _pairs(self, theta):
        th = as_list(theta)
        if len(th) != len(self._chans):
            raise ValueError(f"{len(th)} theta for {len(self._chans)} spin channels")
        return zip(self._chans, th)

    def compute_one_body(self, theta):
        return sum(ch.mult * ch.one_body_expression(th) for ch, th in self._pairs(theta))

    def compute_hartree(self, theta):
        """E_H = 1/2 sum_g rho_g rho_g^+ with rho = sum over the spins of P (and Q for rho^+); closed shell 2 sum P Q.
        Same two contractions as the force bias (raw L, no extra tensor)."""
        P, Q = 0, 0
        for ch, th in self._pairs(theta):
            p, q = ch.k["pq"](ch.M, th)
            P, Q = P + ch.mult * p, Q + ch.mult * q
        return 0.5 * (P * Q).sum(axis=0)

    def compute_exchange(self, theta):
        """Exchange acts within each spin: E_x = -1/2 sum_sigma X_sigma (closed shell -X)."""
        e = 0
        for ch, th in self._pairs(theta):
            x = self._exchange_direct(ch, th) if self.exchange_mode == "direct" else ch.k["exchange"](ch.G, th)
            e = e + (0.5 * ch.mult) * x
        return -e + self._singularity_correction

    def _exchange_gram(self, theta):
        return sum((0.5 * ch.mult) * ch.k["exchange"](ch.G, th) for ch, th in self._pairs(theta))

    def _exchange_direct(self, ch, theta, max_elements=2 * 10**8):
        """Reference form for one spin channel: sum over every column g, E = sum_g sum_{ij} Y_g[j,i] Z_g[i,j]
        (no Gram precontraction).
        Y_g[(Kj,oj),(Ki,oi)] = sum_b2 L[(Kj,oj),(K2,b2),g] theta[(K2,b2),(Ki,oi)], K2 = kmap^-1[q,Kj],
        Z_g[(Ki,oi),(Kj,oj)] = sum_bp conj(L[(Ki,bp'),(Ki,oi),g]) theta[(kmap[q,Ki],bp),(Kj,oj)]. Walkers are chunked."""
        nk, nb, nmax = self._nk, self._nb, self._nmax
        ne = ch.ne
        be = self._be
        w = theta.shape[0]
        tv = theta.reshape(w, nk, nb, nk, ne)
        chunk = max(1, int(max_elements // (nmax * (nk * ne) ** 2)))
        out = []
        for w0 in range(0, w, chunk):
            tvc = tv[w0:w0 + chunk]
            total = 0
            for q in range(nk):
                A = ch.M[q].reshape(nmax, nk, nb, nb)                                  # g, K2, b2, b1
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
        """P, Q of the total density (sum over the spins); closed shell: those of one spin."""
        P, Q = 0, 0
        for ch, th in self._pairs(theta):
            p, q = ch.k["pq"](ch.M, th)
            P, Q = P + (0.5 * ch.mult) * p, Q + (0.5 * ch.mult) * q
        return P, Q

    def _force_bias(self, theta):
        """(2*ng, nw): [ (P+Q)/2 ; i(P-Q)/2 ] / sqrt(nk) for the mean-field subtracted L, summed over the spins
        with weight mult/2 (closed shell: the value of one spin, the factor 2 of the two spins is in the -2j of
        create_auxiliary_field)."""
        fb = 0
        for ch, th in self._pairs(theta):
            fb = fb + (0.5 * ch.mult) * ch.k["force_bias"](ch.M, self._c, self._c_conj, th)
        return fb

    def _auxiliary_field(self, x):
        """sum_g A_e x_e + A_o x_o as one dense (nb*nk, nb*nk, nw) matrix per spin channel, A_e = (L+L^+)/2,
        A_o = i(L-L^+)/2, built block by block: sector q and column k-point K2 fill the block (kmap[q,K2], K2); no
        dense L_g is formed. The same field x acts on both spins."""
        return unlist([ch.k["aux"](ch.M, self._c, self._c_conj, self._eye, x) for ch in self._chans])

    def create_auxiliary_field(self, config, theta, timers=NULL_TIMERS):
        with timers.stage("random_field") as s:
            random_field = s.done(self.create_random_field(config))
        with timers.stage("force_bias") as s:
            force_bias = -2j * self._sqrt_tau * self._force_bias(theta)
            force_bias = config.backend.where(abs(force_bias) > 1, 0.0, force_bias)
            arg = s.done(contract("gw, gw -> w", random_field - 0.5 * force_bias, force_bias))
        x = random_field - force_bias
        with timers.stage("aux_matrix") as s:
            field = s.done(unlist([1j * self._sqrt_tau * ch.k["aux"](ch.M, self._c, self._c_conj, self._eye, x)
                                   for ch in self._chans]))
        return field, config.backend.exp(arg)

    @property
    def h1(self):
        return unlist([ch.h1 for ch in self._chans])

    @property
    def exp_h1(self):
        return unlist([ch.exp_h1 for ch in self._chans])

    @property
    def exp_h1_half(self):
        return unlist([ch.exp_h1_half for ch in self._chans])


# =========================================================================
#  I/O
# =========================================================================
def obtain_H1(config, filename="H1_svd.npy"):
    """Load per-k H1 and assemble block-diagonal multi-k matrix."""
    # memory-mapped read: only the working-precision copy is held in memory
    h1_per_k = np.array(np.load(os.path.expanduser(filename), mmap_mode="r"), dtype=np.dtype(config.complex_type))
    # Expected shape (num_orb, num_orb, num_k) — what opt/vasp produce.
    if h1_per_k.ndim == 2:
        h1_per_k = h1_per_k[:, :, None]
    h1 = reshape_H1(h1_per_k, config.num_kpoint, config.num_orbital)
    return config.backend.array(h1, dtype=config.complex_type)


def obtain_H2(config, filename="H2_zip.npy"):
    """Load H2 in shape (num_orb*num_k, num_orb*num_k, num_g_total)."""
    h2 = np.array(np.load(os.path.expanduser(filename), mmap_mode="r"), dtype=np.dtype(config.complex_type))
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
    """Load H2 (dense or compact) as a numpy array in the working precision and report the
    layout: returns (array, 'dense' | 'compact')."""
    h2 = np.array(np.load(os.path.expanduser(filename), mmap_mode="r"), dtype=np.dtype(config.complex_type))
    if h2.shape[2] != config.num_g:
        raise ValueError(f"H2 has {h2.shape[2]} columns, NGVEC={config.num_g}")
    layout = "compact" if is_compact_shape(
        h2.shape, config.num_orbital, config.num_kpoint) else "dense"
    return h2, layout


def build_hamiltonian(config, one_body, h2_host, layout, q_list, q_sizes=None, one_body_dn=None, h2_host_dn=None):
    """Hamiltonian for a dense or compact H2 (layout from obtain_H2_host).

    dense   -> Hamiltonian (reference path, dense (nb*nk, nb*nk, ng) tensor)
    compact -> HamiltonianCompact (momentum-conserving blocks only)

    Spin-unrestricted runs (config.spin_polarized) always use HamiltonianCompact; a dense H2 (e.g. a Gamma-point
    supercell, nk = 1) is converted to the compact layout. one_body_dn / h2_host_dn are the spin-down terms in the
    same layout as the spin-up ones.
    """
    if config.spin_polarized or layout == "compact":
        kmap = build_kmap(q_list, config.num_kpoint)
        if q_sizes is None:
            q_sizes = sector_sizes_default(config.num_g, config.num_kpoint)
        if layout != "compact":
            h2_host = dense_to_compact(h2_host, kmap, q_sizes, config.num_orbital)
            if h2_host_dn is not None:
                h2_host_dn = dense_to_compact(h2_host_dn, kmap, q_sizes, config.num_orbital)
        # the compact tensor stays on the host, in the working precision, until setup has built the device tensors
        return HamiltonianCompact(one_body=one_body, two_body=h2_host, kmap=kmap, sizes=q_sizes, q_list=q_list,
                                  one_body_dn=one_body_dn, two_body_dn=h2_host_dn)
    return Hamiltonian(
        one_body=one_body,
        two_body=config.backend.array(h2_host, dtype=config.complex_type),
        q_list=q_list)


def initialize_determinant(config):
    """Block-diagonal multi-k trial determinant and initial walker copies.

    Closed shell: returns (trial, walkers). Spin-unrestricted: the trial is a list with one determinant per spin
    channel (first ne_sigma orbitals of every k-point) and the walkers carry slater_det (up) and slater_det_dn."""
    trials, dets = [], []
    for ne, _ in config.channels:
        single = config.backend.eye(config.num_orbital, ne, dtype=config.float_type)
        trial_det = config.backend.block_diag(*([single] * config.num_kpoint))
        trials.append(trial_det)
        dets.append(config.backend.array(
            config.num_walkers * [config.backend.to_numpy(trial_det)]
        ).astype(config.complex_type))
    walkers = Walkers(
        slater_det=dets[0],
        weights=config.backend.ones(config.num_walkers, dtype=config.complex_type),
    )
    walkers.dets = dets
    return unlist(trials), walkers


# =========================================================================
#  Energy and propagation
# =========================================================================
def biorthogonalize(backend, trial, slater_det):
    inverse_overlap = backend.linalg.inv(trial.T @ slater_det)
    return contract("wpi, wij -> wpj", slater_det, inverse_overlap)


def project_trial(backend, trial, slater_det):
    return backend.linalg.det(trial.T @ slater_det) ** 2


def walker_thetas(config, trial, walkers):
    """theta of every spin channel (a bare array for a closed shell)."""
    return unlist([biorthogonalize(config.backend, t, d) for t, d in zip(as_list(trial), walkers.dets)])


def overlap_trial(config, trial, dets):
    """<Psi_T|Phi> = prod over the spins of det(T^T Phi); closed shell det(T^T Phi)^2."""
    trials = as_list(trial)
    if len(trials) == 1 and config.channels[0][1] == 2:
        return project_trial(config.backend, trials[0], dets[0])
    ov = 1
    for t, d in zip(trials, dets):
        ov = ov * config.backend.linalg.det(t.T @ d)
    return ov


def reortho_walkers(config, walkers, method="Auto"):
    """Re-orthogonalise the determinant of every spin channel separately."""
    walkers.dets = [reortho_qr(config, d, method) for d in walkers.dets]


def measure_energy(config, trial, walkers, hamiltonian, timers=NULL_TIMERS):
    with timers.stage("biorthogonalize") as s:
        th = s.done(walker_thetas(config, trial, walkers))
    with timers.stage("one_body") as s:
        e1 = s.done(hamiltonian.compute_one_body(th))
    with timers.stage("hartree") as s:
        eh = s.done(hamiltonian.compute_hartree(th))
    with timers.stage("exchange") as s:
        ex = s.done(hamiltonian.compute_exchange(th))
    energy = e1 + (eh + ex) / config.num_kpoint
    weighted_energy = energy @ walkers.weights
    sum_weights = config.backend.sum(walkers.weights)
    with timers.stage("allreduce"):
        weighted_energy_global = config.comm.allreduce(weighted_energy)
        sum_weights_global = config.comm.allreduce(sum_weights)
    return (
        weighted_energy_global / sum_weights_global,
        weighted_energy_global,
        sum_weights_global,
    )


def measure_hartree(config, trial, walkers, hamiltonian):
    """Hartree energy averaged over walkers; used to compute h_0 for S2 propagator."""
    th = walker_thetas(config, trial, walkers)
    energy_hartree = hamiltonian.compute_hartree(th)
    energy = energy_hartree / config.num_kpoint
    weighted_energy = energy @ walkers.weights
    sum_weights = config.backend.sum(walkers.weights)
    weighted_energy_global = config.comm.allreduce(weighted_energy)
    sum_weights_global = config.comm.allreduce(sum_weights)
    return weighted_energy_global / sum_weights_global


def measure_components(config, trial, walkers, hamiltonian):
    """Returns (E_one, Hartree, Exchange) — useful for verification."""
    th = walker_thetas(config, trial, walkers)
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


def propagate_walkers(config, trial, walkers, hamiltonian, h_0, e_0, timers=NULL_TIMERS):
    """One imaginary-time step of all walkers.

    h_0 = exp(dtau * H_zero) — scalar applied to the new walker determinant (of every spin channel).
    e_0 is the running energy estimate used in the importance reweighting.
    Spin-unrestricted: the one auxiliary field acts on the determinant of each spin with that spin's L and h1,
    and the weight uses the product of the overlaps of the two spins.
    """
    T = timers
    dets = walkers.dets
    with T.stage("biorthogonalize") as s:
        th = s.done(walker_thetas(config, trial, walkers))
    with T.stage("aux_field") as s:
        h2, importance = s.done(hamiltonian.create_auxiliary_field(config, th, T))
    num_rare_event = 0
    fields = as_list(h2)
    if config.propagator == "Taylor":
        h1s = as_list(hamiltonian.h1)
    elif config.propagator == "S1":
        e1s = as_list(hamiltonian.exp_h1)
    elif config.propagator == "S2":
        e1s = as_list(hamiltonian.exp_h1_half)
    else:
        raise ValueError(f"Unknown propagator '{config.propagator}'")

    new_dets = []
    for s, (det, field) in enumerate(zip(dets, fields)):
        if config.propagator == "Taylor":
            with T.stage("taylor") as t:
                full = t.done(apply_taylor(config, h1s[s] + field, det))
            new_dets.append(h_0 * full)
        elif config.propagator == "S1":
            with T.stage("taylor") as t:
                full_h2 = t.done(apply_taylor(config, field, det))
            with T.stage("exp_h1") as t:
                new_dets.append(h_0 * t.done(e1s[s] @ full_h2))
        else:
            with T.stage("exp_h1") as t:
                half = t.done(e1s[s] @ det)
            with T.stage("taylor") as t:
                full_h2 = t.done(apply_taylor(config, field, half))
            with T.stage("exp_h1") as t:
                half = t.done(e1s[s] @ full_h2)
            new_dets.append(h_0 * half)

    new_walkers = Walkers(new_dets[0], config.backend.zeros_like(walkers.weights))
    new_walkers.dets = new_dets
    with T.stage("overlap_weights"):
        new_overlap = overlap_trial(config, trial, new_dets)
        old_overlap = overlap_trial(config, trial, dets)
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


def cholesky_orthonormalize(backend, walkers):
    """Orthonormalize walker columns via Cholesky with the library of the backend (numpy, cupy).

    walkers: [batch, m, n]. Q = A (L^H)^{-1} with A^H A = L L^H, so Q^H Q = I; the dtype of A is kept.
    """
    gram = backend.swapaxes(walkers.conj(), -1, -2) @ walkers
    lh = backend.swapaxes(backend.linalg.cholesky(gram).conj(), -1, -2)
    eye = backend.broadcast_to(backend.eye(lh.shape[-1], dtype=walkers.dtype), lh.shape)
    return walkers @ backend.linalg.solve(lh, eye)


def reortho_qr(config, walker_matrix, method="Auto"):
    """Reorthogonalize walkers.

    method "QR"       : QR of the backend (numpy / jax.numpy / cupy).
    method "Cholesky" : Cholesky of the Gram matrix, done by the library of the backend (JAX, numpy, cupy).
    method "Auto"     : Cholesky on the JAX backend, QR otherwise.
    """
    if method == "Auto":
        method = "Cholesky" if isinstance(config.backend, JaxBackend) else "QR"
    if method == "Cholesky":
        if isinstance(config.backend, JaxBackend):
            return cholesky_orthonormalize_complex_jax(walker_matrix)
        return cholesky_orthonormalize(config.backend, walker_matrix)
    if method != "QR":
        raise ValueError(f"unknown reorthogonalisation method {method!r} (QR, Cholesky or Auto)")
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
    r = config.backend.random_uniform((), dtype=c.dtype) * (W / N)
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


def comb_instances(config, weights):
    """Comb resampling of all walkers: number of copies of every walker (float64 bookkeeping; one uniform number of
    the backend, the same as in every earlier version of the global rebalance)."""
    total_n = len(weights)
    norm_w = weights * (total_n / np.sum(weights))
    bias = -float(config.backend.random_uniform_scalar())
    # running sum bias + w_0 + w_1 + ... in the same order of additions as a loop (identical rounding)
    cum = np.cumsum(np.concatenate(([bias], norm_w)))[1:]
    cur = np.ceil(cum).astype(int)
    return np.diff(cur, prepend=0)


def resample_indices_single(config, walkers_weights):
    """Population control of a run with one MPI process: the same comb as rebalance_global, but only the weights go
    to the host; the caller reorders the walkers on their device with the returned indices (walkers[idx]), so no
    walker is copied to the host and back. Gives exactly the walkers of rebalance_global with one rank."""
    w = np.ascontiguousarray(np.asarray(config.backend.to_numpy(walkers_weights)).real, dtype=np.float64)
    idx = np.repeat(np.arange(len(w)), comb_instances(config, w))
    assert len(idx) == len(w)
    return idx


def rebalance_global(comm, walkers_mats_up, walkers_weights, config):
    """Global rebalance across MPI ranks.

    The walker matrices are gathered and scattered in their own precision (complex64 for Single, complex128 for Double);
    only the weights, which are real here, are handled as float64 for the resampling bookkeeping.
    """
    mats = np.ascontiguousarray(walkers_mats_up)
    local_weights = np.ascontiguousarray(np.asarray(walkers_weights).real, dtype=np.float64)
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_n, n_orb, n_elec = mats.shape
    total_n = local_n * size

    all_weights = np.empty(total_n, dtype=np.float64) if rank == 0 else None
    comm.Gather(local_weights, all_weights, root=0)

    instances = comb_instances(config, all_weights) if rank == 0 else None

    if rank != 0:
        instances = np.empty(total_n, dtype=int)
    comm.Bcast(instances, root=0)

    map_indices = np.repeat(np.arange(total_n), instances)
    assert len(map_indices) == total_n

    all_mats = np.empty((total_n, n_orb, n_elec), dtype=mats.dtype) if rank == 0 else None
    comm.Gather(mats, all_mats, root=0)

    resampled = all_mats[map_indices] if rank == 0 else None
    new_mats = np.empty((local_n, n_orb, n_elec), dtype=mats.dtype)
    comm.Scatter(resampled, new_mats, root=0)
    new_weights = np.ones(local_n, dtype=mats.dtype)
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


def autocorr_error(datastream, c=10.0):
    """Standard error of the mean of a correlated series.

    Uses the integrated autocorrelation time tau_int = 1/2 + sum_{t=1}^{W} rho(t) with the
    automatic window of Sokal (smallest W with W >= c * tau_int(W)); the error is
    sqrt(2 * tau_int * var / N). Unlike the maximum of the blocking curve it does not pick
    up the noisy large-block end of the curve, and unlike a fixed part of the curve it does
    not depend on the block divisor.

    Returns (error, tau_int, window_closed). tau_int is in samples of the series.
    window_closed is False when the window does not close within a tenth of the series,
    i.e. when the series is too short for its correlation time and the error is not reliable.
    """
    x = np.asarray(datastream, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), False
    d = x - x.mean()
    var = np.var(x, ddof=1)
    if var == 0.0:
        return 0.0, 0.5, True
    f = np.fft.rfft(d, 2 * n)
    rho = np.fft.irfft(f * np.conj(f))[:n]
    rho = rho / rho[0]
    tau = 0.5
    closed = False
    for k in range(1, n):
        tau += rho[k]
        if k >= c * tau:
            # a window longer than a tenth of the series is not trustworthy: for a series
            # much slower than its length the sum of rho(t) drifts to zero near the end
            closed = k <= n // 10
            break
    tau = max(tau, 0.5)
    return float(np.sqrt(2.0 * tau * var / n)), float(tau), closed
