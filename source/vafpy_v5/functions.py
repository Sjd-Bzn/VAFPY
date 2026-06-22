"""
vafpy_v5 — differentiable AFQMC for solids (JAX-first, GPU-ready).

This is the Stage-2 prototype from the proposal "Reverse-Mode AD AFQMC
for Forces, Stresses, and Phonons in Solids":

    Stage 0 (molecular AD-AFQMC seed):   covered by Mahajan+Kurian papers
    Stage 1 (periodic energy baseline):  done in vafpy_v3 / vafpy_v4
    Stage 2 (dE/dh, dE/dL):              THIS FILE
    Stage 3 (ionic forces dE/dR):        contract Stage-2 result with dh/dR
    Stage 4 (stress tensor):             contract with strain derivatives

Scope of this prototype (intentionally narrow):
  * num_k = 1 only (Gamma point). Data layout is k-point-aware so adding
    num_k > 1 later does not require restructuring (Q-list / compressed
    H2 paths from v4 will be ported in v5.1).
  * Frozen trial (no coupled-perturbed HF). Proposal Sec. 4.3 level 1.
  * No stochastic reconfiguration during the differentiated trajectory
    — proposal Sec. 8.2 flags SR as a known AD-bias source.
  * No MPI. Single-process JAX. Walkers vectorised on the walker axis.

What works:
    energy = afqmc_energy_path(h1, h2, trial, sd0, w0, key, cfg)
    dE_dh1 = jax.grad(afqmc_energy_path, argnums=0)(h1, h2, ...)
    dE_dh2 = jax.grad(afqmc_energy_path, argnums=1)(h1, h2, ...)

These derivatives have the same per-step scaling as the forward pass
(reverse-mode AD), validated against central finite differences in
test_vafpy_v5.py.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

# Float64 is essential: finite-difference checks of dE/dh need >1e-5 precision,
# which float32 cannot deliver. Set this once at import time.
jax.config.update("jax_enable_x64", True)


# =========================================================================
#  Configuration (static; never traced)
# =========================================================================
@dataclass(frozen=True)
class Config:
    """Static AFQMC parameters. Everything here is a Python scalar.

    Tensor inputs (h1, h2, trial, ...) are passed as JAX arrays to the
    pure functions below — keeping config separate lets `jax.jit` mark
    it as a static argument.
    """
    num_walkers: int
    num_orbital: int        # nb (single-k basis size; num_k=1 here)
    num_electron: int       # ne (per spin; restricted closed-shell)
    num_g: int              # number of Cholesky/SVD channels at this Q
    timestep: float
    num_steps: int
    order_propagation: int = 6
    propagator: str = "S2"   # "S1" or "S2"
    equilibration_frac: float = 0.0  # fraction of leading steps to discard
    fb_cutoff: float = 1.0   # |fb| > cutoff → fb=0 (matches v4 behaviour)
    weight_cap: float = 10.0 # factor > cap → drop walker (matches v4)


# =========================================================================
#  Hamiltonian setup — pure function of (h1, h2)
# =========================================================================
def setup_hamiltonian(h1, h2, trial_single, num_e, dtau):
    """Build the propagation / energy tensors used by the AFQMC loop.

    All operations are pure JAX and differentiable w.r.t. h1 and h2.

    Convention matches v4 at num_k=1:
        h1            : (nb, nb)
        h2            : (nb, nb, ng)        Cholesky / SVD factors
        trial_single  : (nb, ne)            occupied trial orbitals

    Returns a dict of tensors holding the mean-field-subtracted h1_total,
    the precomputed exp(-dtau h1_total) propagator, the H_zero constant,
    the (e, o) split h2 tensors, and the α / α_T / α_mf / α_mf_T trial
    contractions used in force-bias and Hartree/exchange evaluation.
    """
    nb = h2.shape[0]
    occ = jnp.arange(num_e)

    # L_0[g] = sum over occupied i of <i| L^g |i>  (num_k=1)
    L_0 = jnp.sum(h2[occ, occ, :], axis=0)

    h2_dag = jnp.conj(jnp.transpose(h2, (1, 0, 2)))  # h2†

    # Mean-field correction to one-body Hamiltonian.
    # Single (K1=K2=Q=1) block at num_k=1; reproduces v4.H_1_mf exactly.
    avg_A     = 2.0 * L_0
    avg_A_dag = 2.0 * jnp.conj(L_0)
    change = (jnp.einsum("g,rpg->rp", avg_A_dag, h2)
              + jnp.einsum("g,rpg->rp", avg_A, h2_dag))
    h_mf  = h1 + change / 2.0
    h_sic = -jnp.einsum("ijg,kjg->ik", h2, h2_dag) / 2.0
    h1_total = h_mf + h_sic

    # H_zero constant for the S2 propagator's per-step prefactor.
    H_zero = 2.0 * jnp.vdot(L_0, L_0) / (2.0 * num_e)  # vdot conjugates first arg

    # Mean-field-subtracted h2:
    #   h2_mf[p, r, g] = h2[p, r, g] - δ_{pr} * L_0[g] / num_e
    avg_per_elem = avg_A / (2.0 * num_e)              # = L_0 / num_e
    delta = jnp.eye(nb)[:, :, None] * avg_per_elem[None, None, :]
    h2_mf     = h2 - delta
    h2_mf_dag = jnp.conj(jnp.transpose(h2_mf, (1, 0, 2)))

    # h2_e, h2_o = even/odd parts; never combined explicitly in the field
    # — see _apply_h2_field below which uses the z± identity from v4.
    h2_e = (h2_mf + h2_mf_dag) / 2.0
    h2_o = (h2_mf - h2_mf_dag) * 1j / 2.0

    # Trial-contracted tensors used in energy / force-bias evaluation.
    alpha       = jnp.einsum("pi,prg->irg", trial_single, h2)
    alpha_T     = jnp.einsum("pi,prg->irg", trial_single, h2_dag)
    alpha_mf    = jnp.einsum("pi,prg->irg", trial_single, h2_mf)
    alpha_mf_T  = jnp.einsum("pi,prg->irg", trial_single, h2_mf_dag)
    h1_trial    = jnp.einsum("pi,pq->iq", trial_single, h1)

    # Propagators (cached so the scan body avoids the expm).
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
        "h2_e":        h2_e,
        "h2_o":        h2_o,
        "alpha":       alpha,
        "alpha_T":     alpha_T,
        "alpha_mf":    alpha_mf,
        "alpha_mf_T":  alpha_mf_T,
    }


# =========================================================================
#  Energy estimator (mixed estimator; num_k=1)
# =========================================================================
def biorthogonalize(trial, slater_det):
    """theta_w = walker_w @ inv(trial.T @ walker_w).

    trial:       (nb, ne)
    slater_det:  (w, nb, ne)
    returns:     (w, nb, ne)
    """
    inv = jnp.linalg.inv(jnp.einsum("pi,wpj->wij", trial, slater_det))
    return jnp.einsum("wpj,wji->wpi", slater_det, inv)


def compute_energy_per_walker(setup, theta):
    """Mixed-estimator <Ψ_T|H|Φ_w>/<Ψ_T|Φ_w> per walker (num_k=1)."""
    e1 = 2.0 * jnp.einsum("ip,wpi->w", setup["h1_trial"], theta)
    fb   = jnp.einsum("wri,irg->gw", theta, setup["alpha"])
    fb_T = jnp.einsum("wri,irg->gw", theta, setup["alpha_T"])
    e_h  = 2.0 * jnp.einsum("gw,gw->w", fb, fb_T)
    e_x  = -jnp.einsum("wri,jrg,wpj,ipg->w",
                       theta, setup["alpha_T"], theta, setup["alpha"])
    return e1 + e_h + e_x


# =========================================================================
#  One imaginary-time step (S1 or S2). Pure JAX.
# =========================================================================
def _apply_h2_field(setup, slater_det, x_e, x_o):
    """Apply 1j*sqrt_tau * (h2_e·x_e + h2_o·x_o) once to walker.

    Returns the same shape as slater_det. Equivalent to v4's apply_h2_compressed
    at num_k=1, but written as plain einsum (no Q-gather logic needed).
    """
    # h2_op[p, r, w] = h2_e[p, r, g] x_e[g, w] + h2_o[p, r, g] x_o[g, w]
    h2_op = (jnp.einsum("prg,gw->prw", setup["h2_e"], x_e)
             + jnp.einsum("prg,gw->prw", setup["h2_o"], x_o))
    h2_op = 1j * setup["sqrt_tau"] * h2_op
    return jnp.einsum("prw,wri->wpi", h2_op, slater_det), h2_op


def _apply_taylor(h2_op, slater_det, order):
    """exp(1j*sqrt_tau*h2_op) @ slater_det via order-N Taylor series."""
    result = slater_det
    addend = slater_det
    for i in range(order):
        addend = jnp.einsum("prw,wri->wpi", h2_op, addend) / (i + 1)
        result = result + addend
    return result


def propagate_step(setup, cfg: Config, trial, slater_det, weights, key, e_offset):
    """One imaginary-time step. Random fields drawn from `key`.

    No stochastic reconfiguration, no QR reortho, no rebalancing inside —
    those are non-AD operations and should be applied OUTSIDE the
    differentiated `lax.scan` (via `lax.stop_gradient` wrappers in a
    later iteration).
    """
    nw, nb, ne = slater_det.shape
    ng = cfg.num_g
    sqrt_tau = setup["sqrt_tau"]

    theta = biorthogonalize(trial, slater_det)

    # Force bias from the mean-field-subtracted alphas (see v4).
    fb_mf   = jnp.einsum("wri,irg->gw", theta, setup["alpha_mf"])
    fb_mf_T = jnp.einsum("wri,irg->gw", theta, setup["alpha_mf_T"])
    fb_e = (fb_mf + fb_mf_T) / 2.0
    fb_o = (fb_mf - fb_mf_T) * 1j / 2.0
    fb   = jnp.concatenate([fb_e, fb_o], axis=0)     # (2*ng, w)
    fb   = -2j * sqrt_tau * fb
    fb   = jnp.where(jnp.abs(fb) > cfg.fb_cutoff, 0.0, fb)

    # Pathwise random field — same key → same draw across perturbations,
    # which is what we want for an AD/finite-difference check.
    rfield = jax.random.normal(key, (2 * ng, nw), dtype=jnp.float64)
    arg = jnp.einsum("gw,gw->w", rfield - 0.5 * fb, fb)
    importance = jnp.exp(arg)

    shifted = rfield - fb
    x_e = shifted[:ng]
    x_o = shifted[ng:]

    # exp(1j*sqrt_tau h2) sandwiched by exp(-dtau h1) (S1 / S2).
    if cfg.propagator == "S2":
        sd_half = jnp.einsum("pr,wri->wpi", setup["exp_h1_half"], slater_det)
        _, h2_op = _apply_h2_field(setup, sd_half, x_e, x_o)
        sd_mid = _apply_taylor(h2_op, sd_half, cfg.order_propagation)
        sd_new = jnp.einsum("pr,wri->wpi", setup["exp_h1_half"], sd_mid)
    elif cfg.propagator == "S1":
        _, h2_op = _apply_h2_field(setup, slater_det, x_e, x_o)
        sd_mid = _apply_taylor(h2_op, slater_det, cfg.order_propagation)
        sd_new = jnp.einsum("pr,wri->wpi", setup["exp_h1"], sd_mid)
    else:
        raise ValueError(f"Unknown propagator '{cfg.propagator}'")

    sd_new = setup["h_0"] * sd_new

    # Weight update via overlap ratio and importance.
    new_ovr = jnp.linalg.det(jnp.einsum("pi,wpj->wij", trial, sd_new)) ** 2
    old_ovr = jnp.linalg.det(jnp.einsum("pi,wpj->wij", trial, slater_det)) ** 2
    overlap_ratio = new_ovr / old_ovr
    cos_alpha = jnp.cos(jnp.angle(overlap_ratio))
    factor = jnp.abs(overlap_ratio * importance * jnp.exp(cfg.timestep * e_offset))
    factor = jnp.where(factor < cfg.weight_cap, factor, 0.0)
    new_w = factor * jnp.maximum(0.0, cos_alpha) * weights

    return sd_new, new_w


# =========================================================================
#  Energy along a trajectory — THE SCALAR WE DIFFERENTIATE
# =========================================================================
@partial(jax.jit, static_argnames=("cfg",))
def afqmc_energy_path(h1, h2, trial, slater_det_init, weights_init, key, cfg: Config):
    """Run num_steps of AFQMC and return the time-averaged mixed energy.

    Differentiating this w.r.t. h1 (argnums=0) or h2 (argnums=1) yields
    the Stage-2 quantity from the proposal:
        ∂E/∂h_{pq}      and      ∂E/∂L^g_{pq}.
    The same machinery, contracted with ∂h/∂R and ∂L/∂R, gives Stage-3
    ionic forces — but those integral derivatives are out of scope here.
    """
    setup = setup_hamiltonian(h1, h2, trial, cfg.num_electron, cfg.timestep)

    # Initial mixed-estimator energy (used as the trial-energy offset).
    theta0 = biorthogonalize(trial, slater_det_init)
    e_per_w0 = compute_energy_per_walker(setup, theta0)
    e0 = jnp.sum(e_per_w0 * weights_init) / jnp.sum(weights_init)

    def step_fn(carry, key_i):
        sd, w, e_running = carry
        sd_new, w_new = propagate_step(setup, cfg, trial, sd, w, key_i, e_running.real)
        theta_new = biorthogonalize(trial, sd_new)
        e_per_w   = compute_energy_per_walker(setup, theta_new)
        # Guard against the (rare) all-zero-weight case to avoid NaN gradients.
        sum_w = jnp.sum(w_new)
        e_mix = jnp.where(jnp.abs(sum_w) > 1e-30,
                          jnp.sum(e_per_w * w_new) / sum_w,
                          e_running)
        return (sd_new, w_new, e_mix), e_mix

    keys = jax.random.split(key, cfg.num_steps)
    init = (slater_det_init, weights_init, e0)
    (sd_f, w_f, _), e_traj = jax.lax.scan(step_fn, init, keys)

    n_eq = int(cfg.equilibration_frac * cfg.num_steps)
    # e_traj is complex; the AFQMC mixed estimator is reported as the real part.
    return jnp.mean(e_traj[n_eq:].real)


# =========================================================================
#  Convenience: data loading at num_k = 1
# =========================================================================
def load_hamiltonian_data(h1_file="H1_svd.npy", h2_file="H2_zip.npy"):
    """Load v4-format Hamiltonian data and return JAX arrays for num_k=1.

    H1_svd.npy is stored as (nb, nb, num_k) in the legacy pipeline;
    H2_zip.npy is (nb, nb, ng) at num_k=1.
    """
    import numpy as np
    h1 = np.load(h1_file).astype(np.complex128)
    if h1.ndim == 3:
        if h1.shape[2] != 1:
            raise ValueError(
                f"vafpy_v5 prototype is num_k=1; got H1 with num_k={h1.shape[2]}. "
                "K-point support lands in v5.1."
            )
        h1 = h1[:, :, 0]
    h2 = np.load(h2_file).astype(np.complex128)
    return jnp.asarray(h1), jnp.asarray(h2)


def initial_walkers(cfg: Config):
    """Trial Slater det = first `ne` columns of I, walkers initialised to trial."""
    trial = jnp.eye(cfg.num_orbital, cfg.num_electron, dtype=jnp.complex128)
    sd0 = jnp.broadcast_to(trial[None, :, :],
                           (cfg.num_walkers, cfg.num_orbital, cfg.num_electron))
    sd0 = jnp.asarray(sd0, dtype=jnp.complex128)
    w0 = jnp.ones(cfg.num_walkers, dtype=jnp.complex128)
    return trial, sd0, w0


__all__ = [
    "Config",
    "setup_hamiltonian",
    "biorthogonalize",
    "compute_energy_per_walker",
    "propagate_step",
    "afqmc_energy_path",
    "load_hamiltonian_data",
    "initial_walkers",
]
