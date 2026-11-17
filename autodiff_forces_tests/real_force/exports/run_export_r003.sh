#!/bin/bash
# Export at r003 (atom 2 at +0.03 A) and +-0.01 A around it, from the r003 orbitals (r003/vasp/WAVECAR).
module purge; module load vasp-deps
export AOCL_ROOT=/fs/home/sajad/aocl/5.2.0/gcc LD_LIBRARY_PATH=/fs/home/sajad/aocl/5.2.0/gcc/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
vaspexp=/fs/home/sajad/projects/git/vasp_build_spinfix/bin_prim/vasp_ncl
cd "$(dirname "$0")"/r003
for t in c0 p01 m01; do
  (cd $t && md5sum WAVECAR > wavecar.md5 && taskset -c 16 mpirun -np 1 --bind-to none $vaspexp > stdout 2>&1; echo "$t rc=$? $(date)")
done
echo EXPORT_DONE
