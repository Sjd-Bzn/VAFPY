#!/bin/bash
# Primitive C, atom 2 moved by 0.03 A along x (POSCAR), Gamma only, 16 orbitals: same stage sequence and INCARs as
# supper_cell/C_primitive_kpoints/primitive_vasp_p8/run_vasp.sh; differences: KPOINTS 1x1x1, NBANDS = 16 for SUBROT and
# the export, export with the compact exporter (vasp_build_spinfix/bin_prim, 1 rank). 8 ranks on cores 16-23.
set -e
module purge
module load vasp-deps
export AOCL_ROOT=/fs/home/sajad/aocl/5.2.0/gcc
export LD_LIBRARY_PATH=$AOCL_ROOT/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
vaspncl=/fs/home/sajad/projects/git/vasp_build/bin/vasp_ncl
vaspexp=/fs/home/sajad/projects/git/vasp_build_spinfix/bin_prim/vasp_ncl
MPI="taskset -c 16-23 mpirun -np 8 --bind-to none"
ENCUT=1000; GW_CUT=$((ENCUT*4)); EGW_SOFT=$GW_CUT; p=16; j=4
cd "$(dirname "$0")"
echo "vasp: $vaspncl  export: $vaspexp"; date; hostname; pwd
stage() { echo "=== $1 start $(date)"; }
stage DFT
cat INCAR.0_base INCAR.1_dft > INCAR; echo "ENCUT = $ENCUT" >> INCAR
$MPI $vaspncl > stdout.1_DFT 2>&1; cp OUTCAR OUTCAR.1_DFT-$j
stage HF
cat INCAR.0_base INCAR.2_hf > INCAR; echo "ENCUT = $ENCUT" >> INCAR
$MPI $vaspncl > stdout.2_hf 2>&1; cp OUTCAR OUTCAR.2_hf-$j
MAX_NB=$(grep "maximum number of plane-waves" OUTCAR.2_hf-$j | sed "s/://g" | awk '{print $5}')
echo "MAX_NB=$MAX_NB"
stage HF_exact
cat INCAR.0_base INCAR.3_hfex > INCAR; echo "ENCUT = $ENCUT" >> INCAR; echo "NBANDS = $MAX_NB" >> INCAR
$MPI $vaspncl > stdout.3_hfex 2>&1; cp OUTCAR OUTCAR.3_hf_ex-$j; cp WAVECAR WAVECAR.3_hf_ex-$j
stage RPA_natural_orbitals
cat INCAR.0_base INCAR.4_rpano > INCAR; echo "ENCUT = $ENCUT" >> INCAR; echo "NBANDS = $MAX_NB" >> INCAR
$MPI $vaspncl > stdout.4_rpano 2>&1; cp IBZKPT IBZKPT_rpa; cp WAVECAR WAVECAR.4_rpano-$j; cp OUTCAR OUTCAR.4_rpano-$j
stage SUBROT
cat INCAR.0_base INCAR.5_subrot > INCAR; echo "ENCUT = $ENCUT" >> INCAR; echo "NBANDS = $p" >> INCAR
$MPI $vaspncl > stdout.5_subrot 2>&1; cp OUTCAR OUTCAR.5_subrot-$j
stage AFQMC_export
cat INCAR.0_base INCAR.7.AFQMC > INCAR
{ echo "NBANDS = $p"; echo "ENCUT = $ENCUT"; echo "ENCUTGW = $GW_CUT"; echo "ENCUTGWSOFT = $EGW_SOFT"; } >> INCAR
taskset -c 16 mpirun -np 1 --bind-to none $vaspexp > stdout.7_afqmc 2>&1; cp IBZKPT IBZKPT_afqmc; cp OUTCAR OUTCAR.7_afqmc-$j
echo "=== ALL DONE $(date)"
