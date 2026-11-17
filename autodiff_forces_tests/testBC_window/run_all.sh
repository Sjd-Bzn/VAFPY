#!/bin/bash
# Tests B and C on ne: 8 independent chains (cores 16-23), 100 walkers, windows of 100 blocks x 10 steps
# (dE/dlam after 10, 20, ..., 1000 steps), 150 windows each, equilibration 2000 steps.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_window.py $c 100 150 100 10 win_$c.npz > win_$c.log 2>&1 &
done
wait
echo "WINDOWS_DONE $(date)"
