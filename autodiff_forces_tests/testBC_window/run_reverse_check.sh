#!/bin/bash
# Reverse mode (plain and checkpointed) against forward mode, Martin's rule, on the 60 windows of run_eq_window.sh;
# 8 chains on ne cores 16-23, 100 walkers.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=/data/sajad/calcs/autodiff_forces/testBC_reverse
mkdir -p $O
cd "$(dirname "$0")"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_reverse_check.py $c 100 300 60 40 10 $O/rev_$c.npz > $O/rev_$c.log 2>&1 &
done
wait
echo "REV_DONE $(date)"
