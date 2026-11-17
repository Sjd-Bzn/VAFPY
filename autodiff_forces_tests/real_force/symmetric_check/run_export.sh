#!/bin/bash
# Export of H1/H2 at the central geometry (d0) and with atom 2 moved by +-0.01 A along x (p01, m01), starting from the
# orbitals of the central geometry (WAVECAR of C_prim_gamma/vasp) without any SCF stage. Same exporter and INCAR as the
# export stage of C_prim_gamma/vasp/run_vasp.sh, plus ISTART = 1, LWAVE = LCHARG = .FALSE.
module purge; module load vasp-deps
export AOCL_ROOT=/fs/home/sajad/aocl/5.2.0/gcc LD_LIBRARY_PATH=/fs/home/sajad/aocl/5.2.0/gcc/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
vaspexp=/fs/home/sajad/projects/git/vasp_build_spinfix/bin_prim/vasp_ncl
cd "$(dirname "$0")"
for t in d0 p01 m01; do
  (cd $t && md5sum WAVECAR > wavecar.md5 && taskset -c 16 mpirun -np 1 --bind-to none $vaspexp > stdout 2>&1; echo "$t rc=$? $(date)")
done
echo EXPORT_DONE
