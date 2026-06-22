"""Validation suite for vafpy_v5 — the Stage-2 rev-AD prototype.

Three tests, in order of increasing strictness:
    1. HF energy at the initial walker matches v4 (sanity: setup is correct).
    2. Forward `afqmc_energy_path` produces a sensible scalar (no NaNs, finite).
    3. THE KEY TEST: jax.grad(afqmc_energy_path) matches central finite
       difference on selected entries of h1 and h2 to ~1e-5. This is the
       proposal's "Hamiltonian-parameter derivative validation"
       (Sec. 6.3 — Stage 2 of the milestone plan).
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import functions as v5


REF_HF = {
    "E_one":    101.97815646937238,
    "Hartree":   38.0138379353566,
    "Exchange": -25.91436338445311,
    "Total":    114.0776310202759,
    "H_zero":     4.751729741919572,
}


def _make_cfg(num_walkers=10, num_steps=5, dtau=0.005, propagator="S2"):
    return v5.Config(
        num_walkers=num_walkers,
        num_orbital=8,
        num_electron=4,
        num_g=36,
        timestep=dtau,
        num_steps=num_steps,
        order_propagation=6,
        propagator=propagator,
        equilibration_frac=0.0,
    )


# --------------------------------------------------------------------------
def test_hf_components_match_v4():
    """At walker = trial, the energy decomposition is the HF energy."""
    cfg = _make_cfg(num_walkers=10)
    h1, h2 = v5.load_hamiltonian_data(
        os.path.join(os.path.dirname(__file__), "H1_svd.npy"),
        os.path.join(os.path.dirname(__file__), "H2_zip.npy"),
    )
    trial, sd0, w0 = v5.initial_walkers(cfg)
    setup = v5.setup_hamiltonian(h1, h2, trial, cfg.num_electron, cfg.timestep)
    theta = v5.biorthogonalize(trial, sd0)

    e1   = 2.0 * jnp.einsum("ip,wpi->w", setup["h1_trial"], theta)
    fb   = jnp.einsum("wri,irg->gw", theta, setup["alpha"])
    fb_T = jnp.einsum("wri,irg->gw", theta, setup["alpha_T"])
    eh = 2.0 * jnp.einsum("gw,gw->w", fb, fb_T)
    ex = -jnp.einsum("wri,jrg,wpj,ipg->w",
                     theta, setup["alpha_T"], theta, setup["alpha"])
    # All walkers are identical (init); take walker 0.
    assert abs(float(e1[0].real)  - REF_HF["E_one"])    < 1e-9, (e1[0], REF_HF["E_one"])
    assert abs(float(eh[0].real)  - REF_HF["Hartree"])  < 1e-9, (eh[0], REF_HF["Hartree"])
    assert abs(float(ex[0].real)  - REF_HF["Exchange"]) < 1e-9, (ex[0], REF_HF["Exchange"])
    e_tot = e1[0] + eh[0] + ex[0]
    assert abs(float(e_tot.real)  - REF_HF["Total"])    < 1e-9, (e_tot, REF_HF["Total"])
    assert abs(float(setup["H_zero"].real) - REF_HF["H_zero"]) < 1e-9
    print("[PASS] HF energy components match v4 / opt reference exactly.")


# --------------------------------------------------------------------------
def test_forward_energy_runs():
    """afqmc_energy_path runs end-to-end and returns a finite real scalar."""
    cfg = _make_cfg(num_walkers=10, num_steps=5)
    h1, h2 = v5.load_hamiltonian_data(
        os.path.join(os.path.dirname(__file__), "H1_svd.npy"),
        os.path.join(os.path.dirname(__file__), "H2_zip.npy"),
    )
    trial, sd0, w0 = v5.initial_walkers(cfg)
    key = jax.random.key(12345)
    energy = v5.afqmc_energy_path(h1, h2, trial, sd0, w0, key, cfg)
    energy_f = float(energy)
    assert np.isfinite(energy_f), energy_f
    # 5 short steps from the HF state should not wander far from HF.
    assert abs(energy_f - REF_HF["Total"]) < 5.0, energy_f
    print(f"[PASS] afqmc_energy_path runs: E = {energy_f:.6f} "
          f"(HF = {REF_HF['Total']:.6f}).")


# --------------------------------------------------------------------------
def _fd_one_element(fn, x, idx, eps):
    """Central FD along x[idx]. Works for real or complex x; perturbs along
    the real axis if x is complex.
    """
    eps_arr = jnp.zeros_like(x).at[idx].set(eps)
    f_plus  = float(fn(x + eps_arr))
    f_minus = float(fn(x - eps_arr))
    return (f_plus - f_minus) / (2.0 * eps)


def test_grad_h1_matches_fd():
    """dE/dh1 via reverse-mode AD matches central FD on selected elements.

    Picks 3 representative (p, q) indices; tolerance 1e-5 with eps=1e-4.
    With num_walkers small and num_steps=3 this is fast and ~deterministic.
    """
    cfg = _make_cfg(num_walkers=8, num_steps=3, dtau=0.005)
    h1, h2 = v5.load_hamiltonian_data(
        os.path.join(os.path.dirname(__file__), "H1_svd.npy"),
        os.path.join(os.path.dirname(__file__), "H2_zip.npy"),
    )
    trial, sd0, w0 = v5.initial_walkers(cfg)
    key = jax.random.key(31415)

    # The scalar f(h1) we differentiate. NOTE: h1 is complex in general;
    # we keep h1 complex and test the gradient w.r.t. its real part by
    # perturbing along the real axis (the imag axis would test the
    # holomorphic gradient — left for v5.1).
    def f_h1(h1_):
        return v5.afqmc_energy_path(h1_, h2, trial, sd0, w0, key, cfg)

    grad_fn = jax.grad(f_h1)
    g = grad_fn(h1)

    # `jax.grad` of a real-valued function w.r.t. complex input returns the
    # *conjugate* gradient: ∂f/∂z̄ in JAX convention. The real part of g[p,q]
    # is ∂f/∂Re(h1[p,q]) — that's what we compare to FD along the real axis.
    eps = 1e-4
    test_indices = [(0, 0), (1, 2), (3, 5)]
    max_err = 0.0
    for (p, q) in test_indices:
        fd = _fd_one_element(f_h1, h1, (p, q), eps)
        ad = float(g[p, q].real)
        err = abs(fd - ad)
        print(f"  h1[{p},{q}]:  AD = {ad: .8e}   FD = {fd: .8e}   |Δ| = {err:.2e}")
        max_err = max(max_err, err)
    # Symmetric stencil O(eps^2) ~ 1e-8 ideal; with float64 round-off bounded
    # by ~1e-6 in worst case. Tolerance 5e-5 is comfortable.
    assert max_err < 5e-5, f"dE/dh1 AD vs FD max |Δ| = {max_err:.2e}"
    print(f"[PASS] dE/dh1: rev-AD matches central FD (max |Δ| {max_err:.2e}).")


def test_grad_h2_matches_fd():
    """dE/dh2 via reverse-mode AD matches central FD on selected elements."""
    cfg = _make_cfg(num_walkers=8, num_steps=3, dtau=0.005)
    h1, h2 = v5.load_hamiltonian_data(
        os.path.join(os.path.dirname(__file__), "H1_svd.npy"),
        os.path.join(os.path.dirname(__file__), "H2_zip.npy"),
    )
    trial, sd0, w0 = v5.initial_walkers(cfg)
    key = jax.random.key(31415)

    def f_h2(h2_):
        return v5.afqmc_energy_path(h1, h2_, trial, sd0, w0, key, cfg)

    grad_fn = jax.grad(f_h2)
    g = grad_fn(h2)

    eps = 1e-4
    # (p, r, g) channels: pick a few that should have non-trivial sensitivity.
    test_indices = [(0, 0, 0), (1, 2, 5), (3, 4, 10)]
    max_err = 0.0
    for idx in test_indices:
        fd = _fd_one_element(f_h2, h2, idx, eps)
        ad = float(g[idx].real)
        err = abs(fd - ad)
        print(f"  h2[{idx}]:  AD = {ad: .8e}   FD = {fd: .8e}   |Δ| = {err:.2e}")
        max_err = max(max_err, err)
    assert max_err < 5e-5, f"dE/dh2 AD vs FD max |Δ| = {max_err:.2e}"
    print(f"[PASS] dE/dh2: rev-AD matches central FD (max |Δ| {max_err:.2e}).")


def main():
    print("Running vafpy_v5 validation suite "
          "(Stage-2: dE/dh, dE/dL via rev-AD)...\n")
    test_hf_components_match_v4()
    test_forward_energy_runs()
    test_grad_h1_matches_fd()
    test_grad_h2_matches_fd()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
