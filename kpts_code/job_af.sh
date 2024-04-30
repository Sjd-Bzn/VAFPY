#!/bin/bash -l
#SBATCH -J afC648_1                    
#SBATCH -N 1                 
#SBATCH --tasks-per-node=64          
#SBATCH --partition=zen3_1024
#SBATCH --qos p71334_1024

module purge
module load mpi/2021.5.0
export I_MPI_PIN_RESPECT_CPUSET=0
export OMP_NUM_THREADS=64

echo "################ lat_2k ################" 

echo "#############################################"
echo "#############################################"
echo "#############################################"
echo "################### run 1 ###################"
echo "#############################################"
echo "#############################################"
echo "#############################################"
#cp WALKERS_1.npy WALKERS.npy
#cp WEIGHT_1.npy WEIGHT.npy
run_py="python3.6 afqmc_RUN_kpts_1.py"
$run_py
#rm H2_zip_1.npy
mv WALKERS.npy WALKERS_1.npy
mv WEIGHT.npy WEIGHT_1.npy


