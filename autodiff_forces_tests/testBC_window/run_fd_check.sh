#!/bin/bash
# Finite differences (tiny steps, same random numbers) against the forward-mode derivative, rules "off" and "zero";
# first 10 windows of the 8 chains of run_eq_window.sh, ne cores 16-23, 100 walkers.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/data/sajad/calcs/autodiff_forces/testBC_fd
mkdir -p $O
cd "$(dirname "$0")"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_fd_check.py $c 100 300 10 40 10 $O/fd_$c.npz > $O/fd_$c.log 2>&1 &
done
wait
echo "FD_DONE $(date)"
