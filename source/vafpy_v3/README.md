# vafpy_v3: compact-H2 k-point AFQMC

vafpy_v3 is derived from the validated full-H2 implementation (vafpy_v2) and
keeps its numerics. The two-body tensor is stored and used in a compact form
that exploits momentum conservation, so the memory for H2 drops by a factor
`nk` and the dense `(nb*nk, nb*nk, ng*nk)` tensor is never built.

## H2 layouts

```
full     L[(K1,b1), (K2,b2), g]     (nb*nk, nb*nk, ng_total)      vafpy_v2
compact  Lc[(K2,b2), b1, g]         (nb*nk, nb,    ng_total)      vafpy_v3
```

The columns `g` are grouped in sectors of momentum transfer `q`. For a column
k-point `K2` and sector `q` the row k-point is fixed, `K1 = kmap[q, K2]`
(built from `Q_list.npy`), and all other blocks vanish:

    Lc[(K2,b2), b1, g] = L[(kmap[q(g),K2], b1), (K2,b2), g]

The reduced VASP exporter writes this tensor as `(ng*nk, nb, nb*nk)` in numpy
order; the vafpy layout is that array with its axes reversed. After the SVD
every sector keeps a different number of columns, so `Q_sizes.npy` lists the
columns per sector (sum = `NGVEC`). Without it an equal split is assumed.

A dense H2 file is still accepted (layout is detected from the shape) and is
run with the v2 `Hamiltonian` class, which serves as the reference.
For `nk = 1` the two layouts coincide and the file is treated as dense.

## Workflow

1. VASP, reduced exporter (`bin_prim/vasp_ncl`): `H1.npy`, `H2.npy`
   `(ng*nk, nb, nb*nk)`, `k1_list.npy`, `afqmc.yaml`.
2. `Q_list.npy` rows `[K1, K2, Q]` (one-based) from `k1_list`:
   `K1 = k1_list[K2, Q]`.
3. SVD per q sector, same convention and default threshold as
   `compress_H2.py` (columns `U*S`, `s > 1e-4`):

   ```bash
   python3 compress_H2_compact.py --h2 H2.npy --h1 H1.npy --qlist Q_list.npy \
           --num-orb 8 --num-k 8 --threshold 1e-4 --jobs 4 --outdir svd/
   ```

   Writes `H2_zip.npy` `(nb*nk, nb, ng_kept)`, `Q_sizes.npy`, `H1_svd.npy`, and
   prints the retained rank and the reconstruction error of `sum_g L L^+` for
   every sector.
4. Set `NGVEC`, `KPOINT`, `NORB` in `vafpy.in`, then
   `mpirun -n <N> python3 run_kpts.py` (or one process with `BACKEND : JAX`).

## What changed with respect to vafpy_v2

* `HamiltonianCompact` (functions.py): local energy through
  `W[i,r,j,p] = sum_g L[i,r,g] conj(L[p,j,g])` assembled per sector; force bias
  and auxiliary field gathered through `kmap`; mean-field subtraction,
  `h_mf`, `h_sic` and `H_zero` from the diagonal blocks of the sector that
  contains them. Same 1/nk conventions as vafpy_v2.
* `build_hamiltonian` chooses the class from the H2 shape; `QSIZES` input key.
* `compress_H2_compact.py` for the compact SVD.

## Tests

```bash
export AFQMC_DATA=~/projects/data     # test/diamond_kpoint/{supercell_gamma,primitive_k2x2x2,primitive_k2x2x2_compact}
python3 -m pytest .
```

* `test_compact_layout.py`: momentum map, sector bookkeeping, conversions.
* `test_compact_svd.py`: compact SVD (ranks, reconstruction, thresholds).
* `test_compact_hamiltonian.py`: compact vs dense term by term (setup, energy,
  force bias, auxiliary field, propagation) and a same-seed trajectory.
* `test_compact_backends.py`: JAX single precision vs NumPy double (CPU/GPU).
* `test_vafpy_v3_kpoint.py`: primitive 2x2x2 (both layouts) vs the Gamma
  supercell: HF energy, `H_zero`, MP2 normalisation, early decay, rebalance.
* `test_vafpy_v3.py`: inherited single-k checks against the reference code.
