# vafpy_v5.2 — chain-rule forces from rev-AD AFQMC

**Status:** prototype of Stage 3 in the proposal. The chain-rule
contraction is wired up end-to-end; the integral generator (`R → h, L`)
itself is supplied by the user. Two synthetic generators are included
so the plumbing can be validated against finite differences before any
real integral-derivative code is written.

## The chain rule, in code

The proposal Eq. 7:

```
F_I = -dE/dR_I
    = -[ (∂E/∂h)·(∂h/∂R_I) + (∂E/∂L)·(∂L/∂R_I) + ∂E_ion/∂R_I ]
```

In v5.2 this is one JAX call:

```python
def my_integral_fn(R):
    # R : jnp.array of atomic positions
    # Compute h1(R), h2_c(R) from your basis/pseudopotential machinery.
    # Anything JAX-traceable works (PySCF-PBC via custom_vjp, or your
    # own plane-wave structure-factor code).
    return h1, h2_c

force = v52.afqmc_force(R, my_integral_fn, trial, sd0, w0, key, cfg, kpt)
```

JAX's reverse-mode AD threads the gradient backward through your
`integral_fn` and through the AFQMC propagation in a single pass.

## What's verified

| # | Test | Result |
|---|------|--------|
| 1 | R-independent integrals → F = 0 exactly | `max|F| = 0` |
| 2 | Linear `h(R) = h_0 + Σ_α R_α V_α`: rev-AD F matches FD on all components | `Δ ≤ 1e-10` |
| 3 | n_dof = 12 force components in one rev-AD pass | 5× speed-up vs FD on CPU |

Test 2 is the **chain-rule gate**: `(∂E/∂h)·(∂h/∂R)` is computed correctly
across the AFQMC propagation. Test 3 demonstrates the scaling claim from
proposal §3 — many derivatives at constant prefactor.

## Milestone position

| Stage | What | Status |
|-------|------|--------|
| 0 | Molecular AD-AFQMC seed | covered by Mahajan + Kurian papers |
| 1 | Periodic energy baseline | done in v3 / v4 |
| 2 | `∂E/∂h`, `∂E/∂L` via rev-AD | v5 (Γ-only), v5.1 (K-points) |
| **3** | **Forces F = -dE/dR via chain rule** | **v5.2** (synthetic integrals) |
| 3b | Forces with real ∂h/∂R, ∂L/∂R | NOT YET — your integral_fn |
| 4 | Stress tensor (strain derivatives) | not yet |
| 5 | Phonons / elastic constants | not yet |

v5.2 finishes the chain-rule infrastructure. The remaining piece for a
publishable solid-state force benchmark (proposal §6.4, Stage 3) is the
**real** integral_fn — the integral derivatives w.r.t. atomic positions
in your chosen basis.

## When you have real integrals

### Plane-wave + norm-conserving pseudopotentials (Chen-Zhang 2023 route)
The basis is independent of ionic positions, so `∂h/∂R` comes from
differentiating only the structure-factor / pseudopotential pieces.
This is the cleanest route for a first paper.

### Periodic Gaussians via PySCF-PBC
Basis functions depend on R, so `∂h/∂R` includes Pulay terms. Wrap
PySCF integral calls with `jax.custom_vjp` that supplies the analytic
derivative.

### PAW (Taheridehkordi 2023 route)
Augmentation terms, compensation charges, projectors all need
position derivatives. Higher complexity — proposal recommends as a
second-generation implementation.

## Files

- `functions.py` — thin re-export of v5.1 plus the force machinery:
  `afqmc_force`, `afqmc_total_energy_at_R`, `constant_integrals`,
  `linear_perturbation_integrals`.
- `test_vafpy_v5_2.py` — three tests, all passing.
- `H1_svd.npy`, `H2_zip.npy`, `Q_list.npy` — same diamond Γ-point data
  used throughout v3–v5.1.

## What's still deferred (after v5.2)

- **Real `∂h/∂R`, `∂L/∂R`** — depends on which integral back-end the
  user adopts. This is a one-time effort that unlocks all of Stages
  3–5.
- **Stress tensor** — needs `∂h/∂ε`, `∂L/∂ε` (strain derivatives).
  Mechanically the same chain rule; just a different `integral_fn`
  signature (`(R, ε) → (h1, h2_c)`).
- **Trial response** — coupled-perturbed HF response of Ψ_T to R.
  Proposal §4.3 level 2.
- **MPI** — single-process only.

## Caveats

- The integral generator must be **JAX-differentiable**. If your
  integral code lives outside JAX, wrap it with `jax.custom_vjp` and
  supply the analytic Jacobian. Otherwise reverse-mode collapses to a
  finite-difference around the integral call, which negates the
  rev-AD advantage.
- Tests use a **synthetic linear perturbation** of (h1, h2_c). This
  verifies the chain rule but is not physical. Real `V = ∂h/∂R` will
  carry the physics of the integral derivatives.
- `ion_energy_fn` is an optional argument to `afqmc_force` for the
  classical nuclear-nuclear Coulomb piece. Defaults to 0. For a real
  periodic calculation, plug in an Ewald-summed nuclear repulsion.
