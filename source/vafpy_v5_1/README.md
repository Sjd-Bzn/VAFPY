# vafpy_v5.1 — K-points + stabiliser hooks

Extends `vafpy_v5` with the two pieces flagged in the proposal for the
**production** version of the Stage-2 (rev-AD `∂E/∂h`, `∂E/∂L`) prototype:

1. **K-point support.** The compressed-H2 / Q-list infrastructure from v4
   is ported to pure JAX. `num_k = 1` reduces to the v5 path exactly;
   `num_k > 1` enables periodic-cell calculations on Γ-shifted k-meshes
   (the natural setting for diamond / Si / MgO benchmarks).
2. **Stabiliser hooks with detached gradients** — QR reorthogonalisation
   and systematic-resampling rebalance. Both are needed in production
   runs (otherwise walker weights blow up over O(100) steps) but both
   are **discontinuous** w.r.t. h1 / h2, so the proposal §8.2 says they
   must be hidden from the AD pass. Two different strategies are used:
   - **Reortho**: straight-through estimator. Forward applies QR;
     backward acts as identity. Correct to O(walker non-orthogonality).
   - **Rebalance** (systematic resampling): full `lax.stop_gradient`.
     The permutation is irreducibly discontinuous; the AD pass simply
     does not see it. AD vs FD then disagree on rebalance steps — by
     design — and §8.2 asks for a plot of force-bias vs rebalance
     frequency as part of the first paper.

## Milestone position

Same as v5 (Stage 2 of the proposal milestone plan), but with the two
production stabilisers wired up. Forces (Stage 3) and stress (Stage 4)
are still future work and gate on writing `∂h/∂R` / `∂L/∂R` integral
derivatives.

## Files

- `functions.py` — JAX core. `afqmc_energy_path(h1, h2_c, trial, sd0, w0, key, cfg, kpt)`.
- `test_vafpy_v5_1.py` — 7 tests, all passing.
- `H1_svd.npy`, `H2_zip.npy`, `Q_list.npy` — same diamond data, num_k=1.

## What the tests prove

| # | Test | Result |
|---|------|--------|
| 1 | num_k=1 HF energy matches v4 reference | `Δ < 1e-9` |
| 2 | `afqmc_energy_path` runs end-to-end at num_k=1 | finite, near HF |
| 3 | `jax.grad` vs central FD at num_k=1, no stabilisers | `Δ ≤ 9e-11` |
| 4 | Gradient finite with reortho ON (straight-through) | `drift ≈ 2e-3` |
| 5 | Gradient finite with rebalance ON (stop_gradient) | `drift ≈ 1e-2` |
| 6 | num_k=2 HF energy matches num_k=1 per-k-point | `Δ < 1e-13` |
| 7 | `jax.grad` vs FD at num_k=2, no stabilisers | `Δ ≤ 2e-10` |

Test 7 is the **gate** for the K-point port: reverse-mode AD through the
full Q-list / compressed-H2 propagation matches the analytic answer
(via finite difference) to machine precision.

## Public API

```python
import jax
import vafpy_v5_1.functions as v51

# Build everything for diamond at num_k=1 (default heuristic Q-list).
cfg = v51.Config(
    num_walkers=64,
    num_kpoint=1, num_orbital=8, num_electron=4, num_g=36,
    timestep=0.005, num_steps=200,
    propagator="S2",
    reortho_period=5,   # 0 to disable
    rebal_period=0,     # 0 to disable
)
h1, h2_c, kpt = v51.load_hamiltonian_data(
    "H1_svd.npy", "H2_zip.npy", "Q_list.npy",
    num_k=1, num_orb=8, num_e=4, num_g=36,
)
trial, sd0, w0 = v51.initial_walkers(cfg)
key = jax.random.key(0)

# Forward + reverse pass
E      = v51.afqmc_energy_path(h1, h2_c, trial, sd0, w0, key, cfg, kpt)
dE_dh1 = jax.grad(v51.afqmc_energy_path, argnums=0)(h1, h2_c, trial, sd0, w0, key, cfg, kpt)
dE_dh2 = jax.grad(v51.afqmc_energy_path, argnums=1)(h1, h2_c, trial, sd0, w0, key, cfg, kpt)
```

For num_k > 1, pass a `Q_list.npy` matching that k-mesh and update the
shapes accordingly. The default heuristic Q-list (`|K1-K2| == Q-1`) is
used when no `Q_list.npy` is provided.

## What's still deferred

Same as v5 plus:

- **Production-strength K-meshes** — only the default heuristic Q-list
  has been tested. A real periodic calculation will use a Q-list built
  from the actual crystal momentum mesh (consumed via `Q_list.npy`).
- **MPI** — single-process only.
- **Forces / stresses** — need `∂h/∂R`, `∂L/∂R` integral derivatives.

## Caveats / notes

- The QR straight-through estimator is correct only to leading order in
  walker non-orthogonality. For long trajectories where the walker
  deforms significantly between reortho steps, the bias grows. Mahajan
  2023 recommends choosing reortho_period such that this bias is below
  the stochastic error (typically every 5–10 steps for AFQMC). Test 4
  shows ~2e-3 drift after 4 dtau=0.005 steps with reortho every 2 steps
  — well within the proposal's "study as a function of frequency" envelope.
- Systematic-resampling rebalance with stop_gradient yields a biased
  gradient — full stop, by the proposal's own admission. The required
  paper figure (proposal §8.2) is a plot of force/stress error vs
  `rebal_period`.
