"""Derivative of the AFQMC energy with respect to a parameter of the Hamiltonian by automatic differentiation
through the walker propagation (JAX, one process, compact H2).

The Hamiltonian is H1(lam) = H1 + lam * V and, optionally, L(lam) = L + lam * dL for the two-body factors
(compact layout Lc[(K2,b2), b1, g]; H2 = sum_g L_g L_g^+). With dL, every piece built from L (Hartree and exchange
tensors, mean-field shift, mean-field and self-interaction terms of the one-body propagator, H_zero) is rebuilt
from L(lam) inside the differentiated program. The mixed energy estimator
    E(lam) = sum_w W_w E_w / sum_w W_w
is evaluated after every block of steps, and dE/dlam is obtained by differentiating the whole propagation:
walker determinants, force bias, auxiliary field, phaseless weights, re-orthogonalisation and (optionally)
population control. For a converged projection the mixed estimator equals the ground-state energy for every lam,
so dE/dlam is the pure expectation value <V> (Hellmann-Feynman); with V = dH/dR it is a force.

Population control (comb, as in run_kpts) has three modes for the derivative:
  "martin": a copy of walker w gets weight 1 with derivative d log(W_w / sum W) / dlam, i.e.
            dT_ww'/dlam = (1/W_w) (dW_w/dlam) T_ww'  (a walker that survives with 10 % carries 10x its slope)
  "zero":   copies get weight 1 without derivative (dT/dlam = 0; misses the weight derivative)
  "off":    no population control
The comb indices and the random fields do not depend on lam (fixed random numbers): the derivative is the
derivative along one auxiliary-field trajectory. Phase kills and the |a| > 10 cut have zero derivative.

Usage (see test_autodiff_forces.py):
    model = Model(config, hamiltonian, trial, V)          # config with the JAX backend, H set up for the trial
    model = Model(config, hamiltonian, trial, V, L=Lc, dL=dLc)   # two-body perturbation as well
    out = model.window(lam, dets, weights, e0, key, n_blocks, block, rebalance="martin")
    E, dE = model.energy_and_derivative(lam, ...)         # forward mode: dE/dlam for every block
"""
from functools import partial

import numpy as np

import functions as new

jax, jnp = new._try_import_jax()


def expm_taylor(a, order=12, squarings=4):
    """exp(a) by scaling and squaring with a fixed Taylor order: no data-dependent loop, so it can be
    differentiated in forward and reverse mode. For |a| = |dtau H1 / 2| << 1 the error is far below double precision."""
    b = a / (2 ** squarings)
    eye = jnp.eye(a.shape[-1], dtype=a.dtype)
    out, term = eye, eye
    for i in range(1, order + 1):
        term = term @ b / i
        out = out + term
    for _ in range(squarings):
        out = out @ out
    return out


def compact_pieces(Lc, h1, nk, nb, ne, kmap, sizes, dtau):
    """The L-dependent pieces of HamiltonianCompact.setup_energy_expressions (functions.py) as a differentiable JAX
    function of the compact tensor Lc (nb*nk, nb, ng) and the one-body matrix h1 (nb*nk, nb*nk):
    M (padded compact tensor), c (mean-field shift), G (exchange Gram), h1_total (H1 + mean-field change +
    self-interaction), H_zero. Same formulas, same order; the loops are over the static k-point structure."""
    kmap = np.asarray(kmap, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    off = new.sector_offsets(sizes)
    nbk, nek, ng, nmax = nb * nk, ne * nk, int(sizes.sum()), int(sizes.max())
    occ_rows = np.array([K * nb + o for K in range(nk) for o in range(ne)])

    G = []
    for q in range(nk):
        cols = slice(off[q], off[q + 1])
        A = Lc[:, :ne, cols].reshape(nbk * ne, -1)
        B = Lc[occ_rows][:, :, cols].reshape(nek * nb, -1)
        Gq = (A @ B.conj().T).reshape(nk, nb, ne, nk, ne, nb)
        G.append(Gq.transpose(0, 3, 1, 4, 2, 5).reshape(nk, nk, nb * ne, ne * nb))
    G = jnp.stack(G)

    fixed = kmap == np.arange(nk)[None, :]
    mf_sectors = [q for q in range(nk) if fixed[q].any()]
    L0 = jnp.zeros(ng, dtype=Lc.dtype)
    for q in mf_sectors:
        cols = slice(off[q], off[q + 1])
        for K in range(nk):
            L0 = L0.at[cols].add(jnp.einsum("iig->g", Lc[K * nb:K * nb + ne, :ne, cols]))
    H_zero = 2 * jnp.sum(L0 * L0.conj()) / (2 * ne * nk**2)

    change = jnp.zeros((nbk, nbk), dtype=Lc.dtype)
    for q in mf_sectors:
        cols = slice(off[q], off[q + 1])
        avg = 2 * L0[cols]
        for K in range(nk):
            blk = Lc[K * nb:(K + 1) * nb, :, cols]
            change = change.at[K * nb:(K + 1) * nb, K * nb:(K + 1) * nb].add(
                jnp.einsum("prg,g->rp", blk, avg.conj()) + jnp.einsum("rpg,g->rp", blk.conj(), avg))
    h_sic = jnp.zeros((nbk, nbk), dtype=Lc.dtype)
    for q in range(nk):
        cols = slice(off[q], off[q + 1])
        for K2 in range(nk):
            K = kmap[q, K2]
            A = Lc[K2 * nb:(K2 + 1) * nb, :, cols]
            h_sic = h_sic.at[K * nb:(K + 1) * nb, K * nb:(K + 1) * nb].add(jnp.einsum("cbg,cdg->bd", A, A.conj()))
    h1_total = h1 + change / (2 * nk) - h_sic / (2 * nk)

    M = []
    for q in range(nk):
        blockq = Lc[:, :, off[q]:off[q + 1]].reshape(nk * nb * nb, sizes[q]).T
        M.append(jnp.pad(blockq, ((0, nmax - sizes[q]), (0, 0))))
    M = jnp.stack(M)
    c = jnp.zeros(ng, dtype=Lc.dtype)
    for q in mf_sectors:
        c = c.at[off[q]:off[q + 1]].set(L0[off[q]:off[q + 1]] / (ne * nk))
    return dict(M=M, c=c[:, None], cc=c.conj()[:, None], G=G, h1_total=h1_total,
                h0=jnp.exp(dtau * H_zero))


@jax.tree_util.register_pytree_node_class
class Model:
    """Constant pieces of a compact Hamiltonian (set up with the JAX backend for the block-diagonal HF trial)
    and the differentiable propagation. A pytree: the arrays are inputs of the compiled window (not constants
    folded into the program), the sizes and kernels are static."""

    ARRAYS = ("M", "c", "cc", "eye", "G", "sqrt_tau", "h0", "h1_total", "h1_bare", "trial", "V", "L", "dL")
    STATIC = ("dtau", "nk", "order", "ctype", "rtype", "kernels", "sing", "ng", "nw", "nb", "ne", "kmap", "sizes")

    def tree_flatten(self):
        return tuple(getattr(self, a) for a in self.ARRAYS), tuple(getattr(self, s) for s in self.STATIC)

    @classmethod
    def tree_unflatten(cls, static, arrays):
        obj = object.__new__(cls)
        for name, value in zip(cls.ARRAYS, arrays):
            setattr(obj, name, value)
        for name, value in zip(cls.STATIC, static):
            setattr(obj, name, value)
        return obj

    @property
    def k(self):
        return dict(self.kernels)

    def __init__(self, config, hamiltonian, trial, V, L=None, dL=None):
        H = hamiltonian
        if not isinstance(H, new.HamiltonianCompact):
            raise NotImplementedError("the derivative is implemented for the compact H2 layout")
        if not isinstance(config.backend, new.JaxBackend):
            raise NotImplementedError("the derivative needs the JAX backend")
        dt = config.timestep
        self.dtau = dt
        self.nk = config.num_kpoint
        self.order = config.order_propagation
        if config.propagator != "S2":
            raise NotImplementedError("only the S2 propagator is implemented")
        self.ctype = config.complex_type
        self.rtype = config.float_type
        self.kernels = tuple(sorted(H._k.items()))
        self.M, self.c, self.cc, self.eye, self.G = H._M, H._c, H._c_conj, H._eye, H._G
        self.sing = float(H._singularity_correction)
        self.sqrt_tau = jnp.asarray(np.sqrt(dt), dtype=self.rtype)
        self.h0 = jnp.asarray(np.exp(dt * H.H_zero), dtype=self.ctype)
        self.h1_total = jnp.asarray(-np.asarray(H._h1) / dt, dtype=self.ctype)   # H1 + mean-field terms
        self.h1_bare = jnp.asarray(H.one_body, dtype=self.ctype)                 # H1 of the energy
        self.trial = jnp.asarray(trial)
        self.V = jnp.asarray(V, dtype=self.ctype)
        self.ng = config.num_g
        self.nw = config.num_walkers
        self.nb, self.ne = config.num_orbital, config.num_electron
        self.kmap = tuple(map(tuple, np.asarray(H.kmap, dtype=np.int64).tolist()))
        self.sizes = tuple(np.asarray(H.sizes, dtype=np.int64).tolist())
        if (L is None) != (dL is None):
            raise ValueError("a two-body perturbation needs both L (the unperturbed compact tensor) and dL")
        self.L = None if L is None else jnp.asarray(L, dtype=self.ctype)
        self.dL = None if dL is None else jnp.asarray(dL, dtype=self.ctype)
        if dL is not None and self.dL.shape != (self.nb * self.nk, self.nb, self.ng):
            raise ValueError(f"dL has shape {self.dL.shape}, expected {(self.nb * self.nk, self.nb, self.ng)}")

    def pieces(self, lam):
        """Hamiltonian pieces at lam. Without dL only the one-body matrices depend on lam; with dL everything that
        is built from L is rebuilt from L + lam dL (compact_pieces)."""
        h1_bare = self.h1_bare + lam * self.V
        if self.dL is None:
            return dict(M=self.M, c=self.c, cc=self.cc, G=self.G, h1_total=self.h1_total + lam * self.V,
                        h0=self.h0, h1_bare=h1_bare)
        P = compact_pieces(self.L + lam * self.dL, h1_bare, self.nk, self.nb, self.ne, self.kmap, self.sizes,
                           self.dtau)
        P["h1_bare"] = h1_bare
        return P

    # ---- pieces of one step --------------------------------------------------------------------------------
    def biorthogonalize(self, dets):
        inv = jnp.linalg.inv(self.trial.T @ dets)
        return dets @ inv

    def local_energy(self, lam, dets, P=None):
        """Mixed local energy of every walker for the Hamiltonian at lam (pieces P = self.pieces(lam))."""
        P = self.pieces(lam) if P is None else P
        th = self.biorthogonalize(dets)
        h1t = self.trial.T @ P["h1_bare"]                                        # (nek, nbk)
        e1 = 2 * jnp.einsum("ip,wpi->w", h1t, th)
        eh = self.k["hartree"](P["M"], th)
        ex = -self.k["exchange"](P["G"], th) + self.sing
        return e1 + (eh + ex) / self.nk

    def mixed_energy(self, lam, dets, weights, P=None):
        e = self.local_energy(lam, dets, P)
        return jnp.real(jnp.sum(weights * e) / jnp.sum(weights))

    def overlap(self, dets):
        return jnp.linalg.det(self.trial.T @ dets) ** 2

    def step(self, exp_half, e0, dets, weights, x, P=None):
        """One S2 step (propagate_walkers of functions.py) followed by the Cholesky re-orthogonalisation.
        P: pieces of the Hamiltonian (self.pieces(lam)); default: the unperturbed ones."""
        P = self.pieces(0.0) if P is None else P
        th = self.biorthogonalize(dets)
        fb = -2j * self.sqrt_tau * self.k["force_bias"](P["M"], P["c"], P["cc"], th)
        fb = jnp.where(jnp.abs(fb) > 1, 0.0, fb)
        arg = jnp.sum((x - 0.5 * fb) * fb, axis=0)
        field = 1j * self.sqrt_tau * self.k["aux"](P["M"], P["c"], P["cc"], self.eye, x - fb)
        half = exp_half @ dets
        result, addend = half, half
        for i in range(self.order):
            addend = (field @ addend) / (i + 1)
            result = result + addend
        new_dets = P["h0"] * (exp_half @ result)
        ratio = self.overlap(new_dets) / self.overlap(dets)
        factor = jnp.abs(ratio * jnp.exp(arg) * jnp.exp(self.dtau * e0))
        factor = jnp.where(factor < 10, factor, 0.0)
        new_w = factor * jnp.maximum(0.0, jnp.cos(jnp.angle(ratio))) * weights
        new_dets = new.cholesky_orthonormalize_complex_jax(new_dets)
        return new_dets, new_w.astype(self.ctype)

    def comb(self, weights, u):
        """Comb resampling (comb_instances of functions.py) on the device: walker indices of the new population."""
        n = weights.shape[0]
        w = jax.lax.stop_gradient(jnp.real(weights)).astype(jnp.float64 if self.rtype == np.float64 else jnp.float32)
        norm = w * (n / jnp.sum(w))
        cur = jnp.ceil(jnp.cumsum(jnp.concatenate([-u[None], norm]))[1:]).astype(jnp.int32)
        counts = jnp.diff(cur, prepend=0)
        return jnp.repeat(jnp.arange(n), counts, total_repeat_length=n)

    def rebalance(self, dets, weights, u, mode):
        if mode == "off":
            return dets, weights
        idx = self.comb(weights, u)
        if mode == "martin":
            r = weights / jnp.sum(weights)
            new_w = r[idx] / jax.lax.stop_gradient(r[idx])               # value 1, derivative d log(W/sum W)
        elif mode == "zero":
            new_w = jnp.ones_like(weights)
        else:
            raise ValueError(f"unknown rebalance mode {mode!r} (martin, zero, off)")
        return dets[idx], new_w

    # ---- a window of blocks ---------------------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(6, 7, 8, 9))
    def window(self, lam, dets, weights, e0, key, n_blocks, block, rebalance="martin", checkpoint=False):
        """n_blocks blocks of `block` steps starting from (dets, weights); after every block the mixed energy is
        measured and the population is controlled. Returns (energies per block, final dets, final weights).
        The random numbers depend only on key, block and step, not on lam."""
        P = self.pieces(lam)
        exp_half = expm_taylor(-0.5 * self.dtau * P["h1_total"])
        e0 = jax.lax.stop_gradient(e0)

        def one_step(carry, t):
            d, w, kb = carry
            x = jax.random.normal(jax.random.fold_in(kb, t), (2 * self.ng, self.nw), dtype=self.rtype)
            d, w = self.step(exp_half, e0, d, w, x, P)
            return (d, w, kb), None

        if checkpoint:
            one_step = jax.checkpoint(one_step)

        def one_block(carry, b):
            d, w = carry
            kb = jax.random.fold_in(key, b)
            (d, w, _), _ = jax.lax.scan(one_step, (d, w, kb), jnp.arange(block))
            e = self.mixed_energy(lam, d, w, P)
            u = jax.random.uniform(jax.random.fold_in(kb, block + 1), (), dtype=self.rtype)
            d, w = self.rebalance(d, w, u, rebalance)
            return (d, w), e

        (d, w), energies = jax.lax.scan(one_block, (dets, weights), jnp.arange(n_blocks))
        return energies, d, w

    # ---- derivatives ---------------------------------------------------------------------------------------
    def energy_and_derivative(self, lam, dets, weights, e0, key, n_blocks, block, rebalance="martin"):
        """Forward mode: energies of every block and their derivatives dE_b/dlam (one pass)."""
        f = lambda l: self.window(l, dets, weights, e0, key, n_blocks, block, rebalance)[0]
        return jax.jvp(f, (jnp.asarray(lam, dtype=self.rtype),), (jnp.asarray(1.0, dtype=self.rtype),))

    def mean_energy_gradient(self, lam, dets, weights, e0, key, n_blocks, block, rebalance="martin",
                             checkpoint=False):
        """Reverse mode: mean energy over the blocks and its derivative (stores the trajectory; checkpoint=True
        recomputes every step in the backward pass instead of storing its intermediates)."""
        f = lambda l: jnp.mean(self.window(l, dets, weights, e0, key, n_blocks, block, rebalance, checkpoint)[0])
        return jax.value_and_grad(f)(jnp.asarray(lam, dtype=self.rtype))


def hf_derivative(trial, V):
    """dE_HF/dlam for walkers equal to the trial: 2 sum over occupied orbitals of V (closed shell)."""
    t = np.asarray(trial)
    return float(np.real(2 * np.trace(t.T @ np.asarray(V) @ t)))
