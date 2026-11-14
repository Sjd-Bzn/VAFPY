"""Derivative of the AFQMC energy with respect to a parameter of the Hamiltonian by automatic differentiation
through the walker propagation (JAX, one process, compact H2).

The one-body Hamiltonian is H1(lam) = H1 + lam * V. The mixed energy estimator
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


@jax.tree_util.register_pytree_node_class
class Model:
    """Constant pieces of a compact Hamiltonian (set up with the JAX backend for the block-diagonal HF trial)
    and the differentiable propagation. A pytree: the arrays are inputs of the compiled window (not constants
    folded into the program), the sizes and kernels are static."""

    ARRAYS = ("M", "c", "cc", "eye", "G", "sqrt_tau", "h0", "h1_total", "h1_bare", "trial", "V")
    STATIC = ("dtau", "nk", "order", "ctype", "rtype", "kernels", "sing", "ng", "nw")

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

    def __init__(self, config, hamiltonian, trial, V):
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

    # ---- pieces of one step --------------------------------------------------------------------------------
    def biorthogonalize(self, dets):
        inv = jnp.linalg.inv(self.trial.T @ dets)
        return dets @ inv

    def local_energy(self, lam, dets):
        """Mixed local energy of every walker for H1 + lam V."""
        th = self.biorthogonalize(dets)
        h1t = self.trial.T @ (self.h1_bare + lam * self.V)                       # (nek, nbk)
        e1 = 2 * jnp.einsum("ip,wpi->w", h1t, th)
        eh = self.k["hartree"](self.M, th)
        ex = -self.k["exchange"](self.G, th) + self.sing
        return e1 + (eh + ex) / self.nk

    def mixed_energy(self, lam, dets, weights):
        e = self.local_energy(lam, dets)
        return jnp.real(jnp.sum(weights * e) / jnp.sum(weights))

    def overlap(self, dets):
        return jnp.linalg.det(self.trial.T @ dets) ** 2

    def step(self, exp_half, e0, dets, weights, x):
        """One S2 step (propagate_walkers of functions.py) followed by the Cholesky re-orthogonalisation."""
        th = self.biorthogonalize(dets)
        fb = -2j * self.sqrt_tau * self.k["force_bias"](self.M, self.c, self.cc, th)
        fb = jnp.where(jnp.abs(fb) > 1, 0.0, fb)
        arg = jnp.sum((x - 0.5 * fb) * fb, axis=0)
        field = 1j * self.sqrt_tau * self.k["aux"](self.M, self.c, self.cc, self.eye, x - fb)
        half = exp_half @ dets
        result, addend = half, half
        for i in range(self.order):
            addend = (field @ addend) / (i + 1)
            result = result + addend
        new_dets = self.h0 * (exp_half @ result)
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
        exp_half = expm_taylor(-0.5 * self.dtau * (self.h1_total + lam * self.V))
        e0 = jax.lax.stop_gradient(e0)

        def one_step(carry, t):
            d, w, kb = carry
            x = jax.random.normal(jax.random.fold_in(kb, t), (2 * self.ng, self.nw), dtype=self.rtype)
            d, w = self.step(exp_half, e0, d, w, x)
            return (d, w, kb), None

        if checkpoint:
            one_step = jax.checkpoint(one_step)

        def one_block(carry, b):
            d, w = carry
            kb = jax.random.fold_in(key, b)
            (d, w, _), _ = jax.lax.scan(one_step, (d, w, kb), jnp.arange(block))
            e = self.mixed_energy(lam, d, w)
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
