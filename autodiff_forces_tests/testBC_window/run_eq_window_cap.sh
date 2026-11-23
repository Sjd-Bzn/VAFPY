#!/bin/bash
# Local energies capped to e0 +- sqrt(2/dtau). Equilibration (3000 steps from HF) and 60 windows of 400 steps per chain; 8 chains on ne cores 16-23, 100 walkers.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/data/sajad/calcs/autodiff_forces/testBC_eq_window_cap
cd "$(dirname "$0")"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_eq_window.py $c 100 300 60 40 10 $O/eqwcap_$c.npz cap > $O/eqwcap_$c.log 2>&1 &
done
wait
echo "EQW_DONE $(date)"
