"""Validation suite for vafpy_v5.2 — chain-rule forces.

Three tests, in order:
    1. Zero-force sanity: with an R-independent integral generator, the
       reverse-mode force is exactly zero. Validates that the chain-rule
       plumbing is wired up to JAX's autodiff.
    2. Linear-perturbation chain rule: with h(R) = h_0 + Σ_α R_α V_α,
       reverse-mode F = -dE/dR matches central finite differences over
       ALL force components in a single reverse pass. Validates the
       (∂E/∂h)·(∂h/∂R) contraction.
    3. Per-pass scaling demonstration: n_dof components emerge from one
       reverse-mode pass at the cost of one forward energy evaluation.

These tests use SYNTHETIC integral generators because real ∂h/∂R / ∂L/∂R
machinery (PySCF-PBC for Gaussians, structure-factor derivatives for
plane waves) is a separate piece of code outside the scope of this
prototype. Once you have such an `integral_fn`, plug it in and the
SAME `afqmc_force` call returns the physical force vector.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import functions as v52


def _cfg(num_walkers=8, num_steps=3, num_k=1, num_g=36, dtau=0.005):
    return v52.Config(
        num_walkers=num_walkers, num_kpoint=num_k,
        num_orbital=8, num_electron=4, num_g=num_g,
        timestep=dtau, num_steps=num_steps,
        order_propagation=6, propagator="S2",
        equilibration_frac=0.0,
        reortho_period=0, rebal_period=0,
    )


def _load(num_k=1, num_g=36):
    here = os.path.dirname(__file__)
    return v52.load_hamiltonian_data(
        os.path.join(here, "H1_svd.npy"),
        os.path.join(here, "H2_zip.npy"),
        os.path.join(here, "Q_list.npy"),
        num_k=num_k, num_orb=8, num_e=4, num_g=num_g,
    )


# --------------------------------------------------------------------------
def test_constant_integrals_give_zero_force():
    """R-independent integrals must yield F = 0 exactly."""
    cfg = _cfg(num_walkers=6, num_steps=3)
    h1, h2_c, kpt = _load()
    trial, sd0, w0 = v52.initial_walkers(cfg)
    key = jax.random.key(7)

    integral_fn = v52.constant_integrals(h1, h2_c)
    R = jnp.array([0.0, 0.1, -0.2, 0.05, 0.3, -0.1])  # 6-vector, irrelevant

    F = v52.afqmc_force(R, integral_fn, trial, sd0, w0, key, cfg, kpt)
    max_abs = float(jnp.max(jnp.abs(F)))
    assert max_abs < 1e-10, f"constant integrals gave non-zero F: max |F| = {max_abs:.2e}"
    print(f"[PASS] constant integrals: F = 0  (max |F| = {max_abs:.2e}).")


# --------------------------------------------------------------------------
def _build_linear_perturbation(h1, h2_c, n_dof, seed=0):
    """Random Hermitian-preserving V_h1, free V_h2_c, both small."""
    rng = np.random.default_rng(seed)
    nb_k = h1.shape[0]
    nb_compressed = h2_c.shape[0]
    nb_axis2 = h2_c.shape[1]
    ng = h2_c.shape[2]
    # V_h1: build Hermitian (V + V†)/2, then scale down so the linear
    # perturbation doesn't push the Hamiltonian into a wildly different regime.
    V_h1_raw = (rng.standard_normal((n_dof, nb_k, nb_k))
                + 1j * rng.standard_normal((n_dof, nb_k, nb_k))) * 0.01
    V_h1 = 0.5 * (V_h1_raw + jnp.conj(jnp.transpose(V_h1_raw, (0, 2, 1))))
    V_h2_c = (rng.standard_normal((n_dof, nb_compressed, nb_axis2, ng))
              + 1j * rng.standard_normal((n_dof, nb_compressed, nb_axis2, ng))) * 0.001
    return jnp.asarray(V_h1), jnp.asarray(V_h2_c)


def test_force_chain_rule_matches_fd():
    """The headline gate: rev-AD force matches central FD on every component."""
    cfg = _cfg(num_walkers=8, num_steps=3)
    h1, h2_c, kpt = _load()
    trial, sd0, w0 = v52.initial_walkers(cfg)
    key = jax.random.key(31415)

    n_dof = 6
    V_h1, V_h2_c = _build_linear_perturbation(h1, h2_c, n_dof, seed=42)
    integral_fn = v52.linear_perturbation_integrals(h1, h2_c, V_h1, V_h2_c)

    # Evaluate at a generic non-zero R so all V components are active.
    R = jnp.array([0.0, 0.05, -0.03, 0.02, -0.04, 0.01])

    F_ad = v52.afqmc_force(R, integral_fn, trial, sd0, w0, key, cfg, kpt)

    # Central FD along each coordinate.
    def E(R_):
        return v52.afqmc_total_energy_at_R(
            R_, integral_fn, trial, sd0, w0, key, cfg, kpt
        )

    eps = 1e-4
    F_fd = np.zeros(n_dof)
    for a in range(n_dof):
        e_a = jnp.zeros(n_dof).at[a].set(eps)
        F_fd[a] = -(float(E(R + e_a)) - float(E(R - e_a))) / (2.0 * eps)

    max_err = 0.0
    for a in range(n_dof):
        ad = float(F_ad[a])
        fd = float(F_fd[a])
        err = abs(ad - fd)
        max_err = max(max_err, err)
        print(f"  F[{a}]:  rev-AD = {ad: .8e}   FD = {fd: .8e}   |Δ| = {err:.2e}")
    assert max_err < 5e-5, f"force chain-rule AD vs FD max |Δ| = {max_err:.2e}"
    print(f"[PASS] linear-perturbation force chain rule: rev-AD matches FD "
          f"(max |Δ| {max_err:.2e}, n_dof = {n_dof}).")


# --------------------------------------------------------------------------
def test_force_vector_in_one_pass():
    """All n_dof force components come from one reverse-mode pass.

    Times one `afqmc_force` call against n_dof forward energy
    evaluations (the cost of a central FD). For small n_dof the
    rev-AD pass is ~1× a forward; for n_dof >> 1 the rev-AD advantage
    grows. The point here is correctness + a back-of-envelope ratio.
    """
    import time

    cfg = _cfg(num_walkers=8, num_steps=3)
    h1, h2_c, kpt = _load()
    trial, sd0, w0 = v52.initial_walkers(cfg)
    key = jax.random.key(0)

    n_dof = 12   # 4 "atoms" × 3 dirs
    V_h1, V_h2_c = _build_linear_perturbation(h1, h2_c, n_dof, seed=1)
    integral_fn = v52.linear_perturbation_integrals(h1, h2_c, V_h1, V_h2_c)
    R = jnp.zeros(n_dof)

    # Warm jit
    _ = v52.afqmc_force(R, integral_fn, trial, sd0, w0, key, cfg, kpt).block_until_ready()
    _ = v52.afqmc_total_energy_at_R(R, integral_fn, trial, sd0, w0, key, cfg, kpt).block_until_ready()

    t0 = time.time()
    F = v52.afqmc_force(R, integral_fn, trial, sd0, w0, key, cfg, kpt).block_until_ready()
    t_grad = time.time() - t0

    t0 = time.time()
    eps = 1e-4
    for a in range(n_dof):
        e_a = jnp.zeros(n_dof).at[a].set(eps)
        v52.afqmc_total_energy_at_R(R + e_a, integral_fn, trial, sd0, w0, key, cfg, kpt).block_until_ready()
        v52.afqmc_total_energy_at_R(R - e_a, integral_fn, trial, sd0, w0, key, cfg, kpt).block_until_ready()
    t_fd = time.time() - t0

    assert F.shape == R.shape
    assert jnp.all(jnp.isfinite(F))
    print(f"  rev-AD force ({n_dof} components, 1 pass):  {t_grad*1e3:.1f} ms")
    print(f"  FD force      ({n_dof} components, {2 * n_dof} passes):  {t_fd*1e3:.1f} ms")
    print(f"  speed-up:                                {t_fd / max(t_grad, 1e-6):.2f}x")
    print(f"[PASS] one reverse-mode pass produces all {n_dof} force components "
          f"with no NaN/Inf.")


def main():
    print("Running vafpy_v5.2 validation suite (chain-rule forces)...\n")
    test_constant_integrals_give_zero_force()
    test_force_chain_rule_matches_fd()
    test_force_vector_in_one_pass()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
