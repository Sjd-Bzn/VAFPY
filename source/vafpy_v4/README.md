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

## Compact kernels

`HamiltonianCompact` works directly on the compact tensor. Notation: `nb`, `ne` bands and occupied orbitals per
k-point, `nk` k-points, `ng` retained columns in total (about `nk * n_q`), `w` walkers; `kmap[q,K2] = K1`.
The tensor is stored once, padded per sector, as `M[q, g, (K2,b2,b1)]`; nothing of size `(nb*nk, nb*nk, ng)` is built.

| kernel | contraction | cost per walker (v2 dense -> v3) |
|---|---|---|
| force bias | `P[g] = sum theta[(K2,b2),(kmap[q,K2],o)] M[q,g,(K2,b2,o)]`, `Q` with the conjugate slice; `(P+Q)/2`, `i(P-Q)/2` | `2 ng nk^2 ne nb` -> `2 ng nk ne nb` |
| auxiliary field | block `(kmap[q,K2], K2)` of the final matrix = `sum_g M[q,g,(K2,b2,b1)] u_g` (and `conj(M) v_g`) | `2 ng nk^2 nb^2` -> `2 ng nk nb^2` |
| Hartree | `2 sum_g P_g Q_g` (same `P`, `Q`, no extra tensor) | `2 ng nk^2 ne nb` -> `2 ng nk ne nb` |
| exchange | `sum_q sum_{K2,Kb} theta[(K2,b2),(Kb,ob)] theta[(kmap[q,Kb],bp),(kmap[q,K2],oa)] G_q[(K2,b2,oa),(Kb,ob,bp)]` with the per-sector Gram `G_q = sum_g Lc conj(Lc)` of the occupied blocks | `nk^4 ne^2 nb^2` -> `nk^3 ne^2 nb^2` |
| mean field, `h_sic`, `H_zero` | slices of the compact blocks of the sector that contains them; mean-field shift applied as an exact scalar | setup only |

`G` has `nk^3 ne^2 nb^2` elements. The dense reference precontracts the same sum into a `(ne*nk, nb*nk, ne*nk, nb*nk)`
tensor with `nk^4 ne^2 nb^2` elements, 7/8 of them zeros for `nk = 8`. `exchange_mode='direct'` evaluates the exchange as a sum over every column `g` without `G` (reference,
slower for small `nk`). The summed one-body field is the only dense H2-dependent object, `(nb*nk, nb*nk)` per walker,
because the walkers are dense in the combined band-k basis.

On a JAX backend the kernels are compiled with `jit`; single precision stays `complex64`/`float32`.
`benchmark_kernels.py` reports per-kernel time and memory.

## Tests

```bash
export AFQMC_DATA=~/projects/data     # test/diamond_kpoint/{supercell_gamma,primitive_k2x2x2,primitive_k2x2x2_compact}
python3 -m pytest .
```

* `test_compact_layout.py`: momentum map, sector bookkeeping, conversions.
* `test_compact_svd.py`: compact SVD (ranks, reconstruction, thresholds).
* `test_compact_hamiltonian.py`: compact vs dense term by term (setup, energy,
  force bias, auxiliary field, propagation) and a same-seed trajectory.
* `test_compact_kernels.py`: mean-field pieces, force bias, auxiliary-field matrix, Hartree, exchange (Gram and
  sum over g), local energy, weights and reorthogonalisation against the dense reference, and the invariant that
  the full H2 is never built (reconstruction disabled, no array of full-H2 size, host allocation below one full H2).
* `test_compact_backends.py`: JAX single precision vs NumPy double (CPU/GPU).
* `test_vafpy_v3_kpoint.py`: primitive 2x2x2 (both layouts) vs the Gamma
  supercell: HF energy, `H_zero`, MP2 normalisation, early decay, rebalance.
* `test_vafpy_v3.py`: inherited single-k checks against the reference code.
