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
        h_sic = -contract("ijG, jkG -> ik", h2_host, h2_dag) / 2
        h1_total = h_mf + h_sic   # H_1_mf already returns h1 + change/2

        # H_zero = sum of mean-field constant from L_0 (used in S2 propagator).
        L_0 = mean_field_diag(h2_host, config.num_electron,
                              config.num_orbital, config.num_kpoint)
        self.H_zero = 2 * np.einsum("g,g->", L_0, L_0.conj()) / (2 * config.num_electron)

        # Two-body after mean-field subtraction, split into Hermitian/anti-Hermitian.
        h2_mf = A_af_MF_sub(trial_single, trial_host, h2_host, ql,
                            config.num_kpoint, config.num_orbital, config.num_electron)
        two_body_e = gen_A_e(h2_mf)
        two_body_o = gen_A_o(h2_mf)
        two_body_eo = np.concatenate((two_body_e, two_body_o), axis=-1)

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
