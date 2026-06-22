"""
vafpy_v5.2 — chain-rule forces from rev-AD AFQMC.

This is the Stage-3 entry point from the proposal:

    F_I = -dE/dR_I
        = -[ (∂E/∂h)·(∂h/∂R_I)  +  (∂E/∂L)·(∂L/∂R_I)  +  ∂E_ion/∂R_I ]
              ↑ v5 / v5.1 give this part via jax.grad
                                   ↑ NEW: the user's integral generator
                                                                ↑ classical, optional

The `(∂E/∂h)(∂h/∂R)` and `(∂E/∂L)(∂L/∂R)` contractions are handled
automatically by JAX as long as the integral generator
`integral_fn: R -> (h1, h2_c)` is itself differentiable. This file
provides:

  * a thin re-export of vafpy_v5.1 (so users can do everything from
    one module), and
  * the force-machinery wrappers `afqmc_force` /
    `afqmc_total_energy_at_R`,
  * two built-in synthetic integral generators
    (`constant_integrals`, `linear_perturbation_integrals`) that let
    the chain-rule plumbing be validated against finite differences
    without yet needing a real PySCF-PBC or plane-wave back-end.

WHEN YOU HAVE REAL ∂h/∂R INTEGRALS:

    Provide a JAX-differentiable callable
        def my_integral_fn(R):
            # R : jnp.array of atomic positions, shape (n_atom, 3) or flat
            # h1 : (nb*nk, nb*nk) complex
            # h2_c : (nb*nk, nb, ng) complex
            return h1, h2_c
    and call
        force_vec = v52.afqmc_force(R, my_integral_fn, ...).
    JAX's reverse-mode AD threads through both your integral code and
    the AFQMC propagation in a single backward pass — that is the
    constant-prefactor scaling promised by the proposal.

  If your integrals come from non-differentiable code (e.g. external
  Fortran), wrap it with a custom_vjp that defines ∂h/∂R analytically.
  See the proposal §11.2 for the Kurian-Mahajan-Sharma molecular
  approach to this.
"""
import importlib.util as _ilu
import os as _os
import sys as _sys

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------
#  Re-export v5.1 — load it as a uniquely-named module so multiple version
#  imports in one process do not collide.
# --------------------------------------------------------------------------
def _load_v51():
    here = _os.path.dirname(_os.path.abspath(__file__))
    v51_path = _os.path.join(here, "..", "vafpy_v5_1", "functions.py")
    spec = _ilu.spec_from_file_location("_vafpy_v51_functions", v51_path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_v51 = _load_v51()

# Public re-exports — anything users built workflows around in v5.1 keeps
# working unchanged. Kept explicit so the API surface is auditable.
Config                    = _v51.Config
KptInfo                   = _v51.KptInfo
build_kpt_info            = _v51.build_kpt_info
compress_h2               = _v51.compress_h2
expand_h2                 = _v51.expand_h2
setup_hamiltonian         = _v51.setup_hamiltonian
biorthogonalize           = _v51.biorthogonalize
compute_energy_per_walker = _v51.compute_energy_per_walker
propagate_step            = _v51.propagate_step
afqmc_energy_path         = _v51.afqmc_energy_path
load_hamiltonian_data     = _v51.load_hamiltonian_data
initial_walkers           = _v51.initial_walkers
reshape_h1_per_k          = _v51.reshape_h1_per_k


# =========================================================================
#  Built-in synthetic integral generators
# =========================================================================
def constant_integrals(h1, h2_c):
    """Returns an integral_fn that ignores R and yields fixed (h1, h2_c).

    Useful as a sanity test: the force from this generator MUST be zero
    to machine precision — there is literally no R dependence anywhere
    in the graph.
    """
    def integral_fn(R):
        del R
        return h1, h2_c
    return integral_fn


def linear_perturbation_integrals(h1_0, h2_c_0, V_h1, V_h2_c):
    """Linear-in-R synthetic generator: h(R) = h_0 + Σ_α R[α] V[α].

    Validates the chain rule end-to-end:
        dE/dR_α = Σ_pq (∂E/∂h_{pq}) V_h1[α, p, q]
                + Σ_prg (∂E/∂h2_c_{prg}) V_h2_c[α, p, r, g]
    The user passes:
        h1_0 : (nb*nk, nb*nk) baseline one-body
        h2_c_0 : (nb*nk, nb, ng) baseline Cholesky factors (compressed)
        V_h1 : (n_dof, nb*nk, nb*nk) per-coordinate perturbation of h1
        V_h2_c : (n_dof, nb*nk, nb, ng) per-coordinate perturbation of h2
    and gets a generator that produces (h1, h2_c) from a flat coordinate
    vector R of length n_dof.

    This is NOT physical — `V` should come from real ∂h/∂R, ∂L/∂R
    integral derivatives. It exists to verify the chain-rule plumbing.
    """
    def integral_fn(R):
        R_cast = R.astype(V_h1.dtype)
        h1 = h1_0 + jnp.einsum("a,apq->pq", R_cast, V_h1)
        R_cast2 = R.astype(V_h2_c.dtype)
        h2_c = h2_c_0 + jnp.einsum("a,aprg->prg", R_cast2, V_h2_c)
        return h1, h2_c
    return integral_fn


# =========================================================================
#  Total energy at R, force vector
# =========================================================================
def afqmc_total_energy_at_R(R, integral_fn, trial, sd0, w0, key,
                            cfg: Config, kpt: KptInfo,
                            ion_energy_fn=None):
    """E_total(R) = E_AFQMC(integral_fn(R)) + E_ion(R).

    `ion_energy_fn(R) -> scalar` is optional (defaults to 0). For a real
    periodic calculation it would be the Ewald sum of nuclear-nuclear
    Coulomb repulsion.
    """
    h1, h2_c = integral_fn(R)
    E_el = afqmc_energy_path(h1, h2_c, trial, sd0, w0, key, cfg, kpt)
    if ion_energy_fn is None:
        return E_el
    return E_el + ion_energy_fn(R)


def afqmc_force(R, integral_fn, trial, sd0, w0, key,
                cfg: Config, kpt: KptInfo,
                ion_energy_fn=None):
    """Force vector F = -dE_total/dR via a single reverse-mode pass.

    Returns an array of the same shape as `R`. Per the proposal §3,
    all force components are obtained at constant prefactor relative
    to one forward energy evaluation — that's the rev-AD payoff.
    """
    def E_total(R_):
        return afqmc_total_energy_at_R(
            R_, integral_fn, trial, sd0, w0, key, cfg, kpt, ion_energy_fn
        )
    return -jax.grad(E_total)(R)


__all__ = [
    # Re-exports from v5.1
    "Config", "KptInfo", "build_kpt_info", "compress_h2", "expand_h2",
    "setup_hamiltonian", "biorthogonalize", "compute_energy_per_walker",
    "propagate_step", "afqmc_energy_path",
    "load_hamiltonian_data", "initial_walkers", "reshape_h1_per_k",
    # New in v5.2
    "constant_integrals", "linear_perturbation_integrals",
    "afqmc_total_energy_at_R", "afqmc_force",
]
