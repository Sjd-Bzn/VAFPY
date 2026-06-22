# vafpy_v4 — compressed-H2 periodic AFQMC

**Status:** production AFQMC energy code, memory-optimised version of v3.
Bit-identical results to v3, lower RAM. This is the **current best
production code for energies**. No automatic differentiation.

## What it does

Everything v3 does, plus:

- **Compressed H2 storage** at runtime: `(nb*nk, nb, ng*nk)` instead of
  v3's `(nb*nk, nb*nk, ng*nk)`. The k2 axis is implicit via momentum
  conservation (`k2 = k2_map[k1, Q]`), saving nk-fold memory.
- **Per-Q gather propagation**: the dense `(nb*nk, nb*nk, nw)` propagator
  matrix is *never materialised*. Each Q-block of h2 is gathered and
  applied on the fly through NumPy advanced indexing.
- **5-D tensor caching** of the `α / α_T / α_mf / α_mf_T` contractions
  for fast per-step force-bias and Hartree / exchange evaluation.

## Memory savings (vs v3)

For a system with `nb=8`, `nk=4`, `ng_total=144`:
- v3 H2 size: 32 × 32 × 144 ≈ 147k complex entries
- v4 H2 size: 32 × 8 × 144  ≈  37k complex entries  (4× smaller)
- The savings grow linearly with `nk`.

## How to run

Same as v3 — drop-in replacement:

```bash
cd vafpy_v4
mpirun -n <N> python3 run_kpts.py
```

Input file (`vafpy.in`) format is identical to v3. Default test data
(diamond Γ-point) is also identical. The compressed-H2 path activates
automatically when `KPOINT > 1`; at `KPOINT=1` the layout collapses to
v3's dense form (so v3 and v4 produce identical output bit-for-bit).

## Tests

```bash
python3 test_vafpy_v4.py    # HF energy, propagator, multi-k consistency
python3 test_v4_vs_v3.py    # bit-identical match to v3 on several inputs
python3 test_backends.py    # NumPy / JAX produce matching HF energies
```

`test_v4_vs_v3.py` is the central correctness gate: v4 must produce
bit-identical HF energy and bit-identical propagated walkers as v3 for
the same inputs (and same fixed random fields).

## Files

- `functions.py` — core AFQMC + compressed-H2 helpers (~1080 lines).
- `run_kpts.py` — MPI driver (identical to v3).
- `input_reader.py`, `vafpy.in` — same as v3.
- `H1_svd.npy`, `H2_zip.npy`, `Q_list.npy` — diamond data.
- `test_*.py` — verification suite including the v3 cross-check.

## What's NOT here

Same as v3: no automatic differentiation, no forces, no response
properties. For those, use v5 (Γ-only), v5.1 (K-points), or v5.2
(force chain rule).

## Relation to other versions

```
v3   →   v4   →   v5   →   v5.1   →   v5.2
dense    compressed   JAX rev-AD   + k-points   + force chain rule
                     (Γ only)     stop_gradient
```

v4 is the **last NumPy/CuPy/JAX-backend energy-only version**. v5.x is a
clean break: pure JAX, functional style, designed for `jax.grad`.
