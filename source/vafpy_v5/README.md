# vafpy_v5 — reverse-mode AD AFQMC (Stage 2)

**Status:** prototype. Implements the Hamiltonian-parameter derivative step
(`∂E/∂h`, `∂E/∂L`) from the proposal *"Reverse-Mode Automatic Differentiation
AFQMC for Forces, Stresses, and Phonons in Solids"*.

This is the **starting** point of the rev-AD project, not the finished
product. It exists to prove that the AFQMC propagation in this codebase
can be differentiated end-to-end via JAX reverse-mode, with the gradient
verified against central finite differences. Once this gate is passed,
the same machinery can be contracted with `∂h/∂R` / `∂L/∂R` (Stage 3,
ionic forces) and `∂h/∂ε` (Stage 4, stress).

## Milestone position (proposal §6)

| Stage | What                                  | Status        |
|-------|---------------------------------------|---------------|
| 0     | Molecular AD-AFQMC seed               | covered by Mahajan+Kurian papers |
| 1     | Periodic energy baseline              | done in v3 / v4 |
| **2** | **`∂E/∂h`, `∂E/∂L` via rev-AD**       | **this prototype** |
| 3     | Ionic forces (contract with ∂h/∂R)    | not yet |
| 4     | Stress tensor                         | not yet |
| 5     | Phonons / elastic constants           | not yet |

## What's in here

- `functions.py` — pure-JAX AFQMC at the Γ-point. The key entry point is
  `afqmc_energy_path(h1, h2, trial, sd0, w0, key, cfg)`, a `jax.jit`-ed
  scalar of `(h1, h2)` ready for `jax.grad`.
- `test_vafpy_v5.py` — four-test validation suite. The decisive test
  compares `jax.grad(afqmc_energy_path, argnums=0)` against central FD
  on selected entries of `h1` and `h2`.
- `H1_svd.npy`, `H2_zip.npy`, `Q_list.npy` — copied from v4 (diamond,
  num_k=1, NORB=8, NEUP=4, NGVEC=36).

## Deliberate restrictions (scope)

The proposal's code-level checklist (§12.1) says: *"separate differentiable
propagation from non-differentiable stochastic reconfiguration"* and
*"first return `∂E/∂h` and `∂E/∂L`; postpone full coordinate derivatives
until these are validated."* v5 follows that literally:

- **num_k = 1 only.** K-point support (compressed H2 / Q-list logic from
  v4) will be ported in v5.1 — the file layout already preserves the
  K-point-aware data inputs so this is a drop-in extension.
- **Frozen trial** (proposal §4.3 level 1). No coupled-perturbed HF.
- **No stochastic reconfiguration**, no QR reortho, no rebalancing inside
  the differentiated trajectory. These are the discontinuous operations
  flagged in proposal §8.2.
- **No MPI.** Single-process JAX. Walkers vectorised on the walker axis.
- **Pathwise random fields** — `jax.random.PRNGKey(seed)` deterministically
  splits into per-step keys, so perturbations of `h1` see the same draws.
- **No `∂h/∂R`, `∂L/∂R`** — the integral-derivative machinery (Pulay-free
  plane-wave route from Chen & Zhang 2023, or Gaussian integral derivatives
  from Kurian et al. 2026) is the next file to write, not this one.

## Running

```bash
cd vafpy_v5
python3 test_vafpy_v5.py
```

Expected output ends with `All tests passed.`. On CPU each test takes
~5–15 s (JIT compile dominates); on GPU the per-step cost is negligible
and only setup time remains.

## How to use the gradients

```python
import jax
import vafpy_v5.functions as v5

cfg = v5.Config(num_walkers=64, num_orbital=8, num_electron=4,
                num_g=36, timestep=0.005, num_steps=200, propagator="S2")
h1, h2 = v5.load_hamiltonian_data("H1_svd.npy", "H2_zip.npy")
trial, sd0, w0 = v5.initial_walkers(cfg)
key = jax.random.key(0)

# Forward energy
E = v5.afqmc_energy_path(h1, h2, trial, sd0, w0, key, cfg)

# Reverse-mode derivatives — same per-step scaling as the forward pass
dE_dh1 = jax.grad(v5.afqmc_energy_path, argnums=0)(h1, h2, trial, sd0, w0, key, cfg)
dE_dh2 = jax.grad(v5.afqmc_energy_path, argnums=1)(h1, h2, trial, sd0, w0, key, cfg)
```

`dE_dh1` has the same shape as `h1`; `dE_dh2` has the same shape as `h2`.
JAX's convention for `grad` of a real-valued scalar w.r.t. a complex
input returns the conjugate Wirtinger derivative; for real-axis
perturbations the real part is the quantity that matches FD.

## What's next (in order of risk)

1. **K-point support** (v5.1) — port the compressed H2 / Q-list paths
   from v4 into the JAX setup_hamiltonian. Add a multi-k test that
   mirrors `test_multi_k_consistency` from v4.
2. **`lax.stop_gradient` around SR / reortho / rebalance** — turn the
   full v4 production loop on, but mark the discontinuous steps as
   non-differentiable. Then study reconfiguration-interval bias on the
   gradient (proposal §8.2 — a required plot for the first paper).
3. **`∂h/∂R`, `∂L/∂R`** — integral derivatives. For a Gaussian periodic
   route, interface through PySCF-PBC; for a plane-wave route, compute
   the structure-factor-style derivatives analytically.
4. **First solid-state force benchmark**: displaced-atom diamond force,
   compared against finite-difference AFQMC (proposal Stage 3).

## Notes for the reviewer (you, future me)

- `jax.scipy.linalg.expm` is differentiable as of JAX 0.4.x. We rely on
  this; if it ever changes, the matrix exponential needs an explicit
  Padé+grad path.
- `jax.lax.scan` stores all per-step intermediates by default → memory
  is O(num_steps × walker_state). For long trajectories use
  `jax.checkpoint` / `jax.remat` to checkpoint every ~√num_steps steps.
  Not needed yet at 100–1000 steps.
- The mixed-estimator energy is reported as `Re E`. The imaginary part
  vanishes in expectation; pulling it back through `jax.grad` would
  introduce noise we don't want.
- All tensor inputs are `complex128`. The dependency on `jax_enable_x64`
  is set at import time in `functions.py`.
