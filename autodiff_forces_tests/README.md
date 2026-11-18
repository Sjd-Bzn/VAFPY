# Tests of the AFQMC derivative (forces by automatic differentiation)

Scripts and inputs of all tests of `source/vafpy_v3/autodiff_forces.py`. The method, the tests and the results are
described in the report `afqmc-forces-guide.pdf` (source `afqmc-forces-guide.tex` in the afqmc-metal_oxides
repository). Large outputs (VASP files, AFQMC samples) are not in the repository; they are in
`/data/sajad/calcs/autodiff_forces` (ne) and `/gpfs/data/fs71334/sajad/autodiff_forces` (VSC5). The scripts contain
the absolute paths used there.

System of all tests: diamond primitive cell (2 atoms), Gamma point, 16 orbitals (4 occupied), from VASP
(DFT -> HF -> HF exact -> RPA natural orbitals -> SUBROT -> export); 100 walkers, time step 2.5e-4 1/eV, comb every
10 steps. Test F also uses the 2x2x2 k-mesh.

## Stage 1: model perturbation V = H1 (dE/dlam = <H1>, HF limit = VASP E1)

| directory | test |
|---|---|
| `exactness/` | the 12 exactness tests of `test_autodiff_forces.py` without pytest (`run_exact.py`), long windows (`long.py`) |
| `testA_bias/` | rebalancing rules (Martin, zero, off), 8 chains x 1000 windows of 40 steps |
| `testBC_window/` | window length (up to 1000 steps), noise; fixed-lambda reference energies; mixed estimator of <H1> (`run_mixed.py`) and figure (`plot_window_mixed.py`) |
| `testFD_validation/` | pathwise finite difference (`run_path.py`) and ensemble finite difference with common random numbers (`run_fd.py`), analyses |
| `testF_memory/` | memory and time against window length, walkers and system (ne CPU, he GPU, VSC5 CPU), `plot_memory.py` |

## Stage 2: real force, atom 2 displaced by 0.03 A along x

| directory | content |
|---|---|
| `real_force/symmetric_check/` | exports at 0 and +-0.01 A around the perfect crystal from its orbitals (zero force by symmetry), `compare.py` |
| `real_force/vasp_r003/` | full VASP pipeline at the displaced geometry (`run_vasp.sh`, `INCAR.*`, `POSCAR`, `KPOINTS`; POTCAR not included) |
| `real_force/exports/` | export-only runs `c0` (+0.03 A, the reference), `p01` (+0.04 A), `m01` (+0.02 A) with the WAVECAR of `vasp_r003` (`INCAR`, `POSCAR`); `run_export_r003.sh`; `force_r003.py` (frozen-orbital HF force against VASP); `hf_check_dHdR.py` (derivative code with the real dH at the HF level) |
| `real_force/svd_joint/` | `compress_joint.py`: one SVD of [L ; s dL], compressed L and dL in the same basis; `hf_check_compressed.py` |
| `real_force/afqmc_force/` | AFQMC force: `common.py` (setup), `run_fd.py` (derivative with Martin's and the zero rule, correlated finite difference at +-h), `analyse.py`, `analyse_paired.py`, `plot_force.py`, `run_pilot.sh`, `run_prod.sh` (ne), `job_force_vsc5.slrm` (VSC5) |
| `results/` | text output of the analyses quoted in the report (`real_force_*`: stage 2; `testFD_*`: stage 1) |

## Order of the steps for a force

1. Full VASP pipeline at the geometry R (`vasp_r003/run_vasp.sh`).
2. For every atom and direction: two export-only runs at R +- delta with the WAVECAR of step 1 (`ISTART = 1`,
   `LWAVE = LCHARG = .FALSE.`, only POSCAR changed).
3. dH1 and dL by central differences; one joint SVD of L and all dL (`svd_joint/compress_joint.py`).
4. Checks at the HF level (`exports/force_r003.py`, `exports/hf_check_dHdR.py`, `svd_joint/hf_check_compressed.py`).
5. AFQMC: equilibrate, then windows of converged length; average dE/dlam; add the Ewald and PAW double-counting
   terms from `afqmc.yaml`.
