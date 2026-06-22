# vafpy_v3 — periodic ph-AFQMC energy (dense H2 runtime)

**Status:** production AFQMC energy code with K-point support and a
multi-backend abstraction (NumPy / JAX / CuPy). No automatic
differentiation. Produces ground-state energies for solids; everything
in v5.x is built on this foundation.

## What it does

- Phaseless auxiliary-field QMC for periodic systems with K-points.
- S1 and S2 imaginary-time propagators (S2 = exp(-h1·dτ/2) · exp(h2) · exp(-h1·dτ/2)).
- Mean-field-subtracted Hamiltonian construction.
- Cross-backend execution: pick `NumPy`, `JAX`, or `CuPy` from the input file.
- MPI parallelism across walker populations.
- Systematic-resampling rebalance + QR reorthogonalisation for walker stability.
- Block averaging + autocorrelation-aware error analysis at the end of the run.

## Architecture

- **Dense H2 storage** at runtime — `(nb*nk, nb*nk, ng*nk)`. Simplest
  but memory-hungry for large k-meshes. (v4 fixes this with compressed
  storage; v5.x rebuilds in pure JAX for AD.)
- One `Hamiltonian` dataclass holds h1, h2, mean-field-derived tensors.
- `Configuration` dataclass holds run-control parameters.
- Module-level functions for propagation, energy, rebalance.

## How to run

```bash
cd vafpy_v3
# Edit vafpy.in (input file — YAML) for system / control parameters
# then:
mpirun -n <N> python3 run_kpts.py
```

The input file (`vafpy.in`) sets system size, walkers, timestep, etc.
See the included file for a working diamond Γ-point example (`NORB=8`,
`NEUP=4`, `NGVEC=36`, `KPOINT=1`, `DTAU=0.005`, `NWAK=100`,
`NSTP=20`). Output: `AFQMC_CS_*.txt` trajectory + `outcar.txt` summary.

### Key input keys (vafpy.in)
- `NORB`, `NEUP`, `NEDOWN`, `NGVEC`, `KPOINT` — system size
- `DTAU`, `NSTP`, `NWAK` — time step, number of steps, walkers per rank
- `PROPAG` — "S1" or "S2"
- `OTEY` — Taylor order for exp(h2)
- `REOPRI`, `REBPRI` — reortho / rebalance frequencies
- `BACKEND` — "NumPy", "JAX", or "CuPy"
- `PRECSION` — "Single" or "Double"

## Tests

```bash
python3 test_vafpy_v3.py        # HF energy + propagator vs opt reference
python3 test_compressed_h2.py   # compressed/dense round-trip + identical results
python3 test_backends.py        # NumPy and JAX produce matching HF energies
```

## Files

- `functions.py` — core AFQMC machinery (~800 lines).
- `run_kpts.py` — MPI driver.
- `input_reader.py` — YAML input parser.
- `vafpy.in` — diamond Γ-point input.
- `H1_svd.npy`, `H2_zip.npy`, `Q_list.npy` — diamond data.
- `test_*.py` — verification suite.

## What's NOT here

- No automatic differentiation. For forces / response properties, use
  v5 / v5.1 / v5.2.
- Memory grows with full `(nb*nk)²` for H2. v4 compresses this.

## Relation to other versions

```
v3   →   v4   →   v5   →   v5.1   →   v5.2
dense    compressed   JAX rev-AD   + k-points   + force chain rule
                     (Γ only)     stop_gradient
```
