#!/bin/bash
# Pilot on ne (cores 16-23): 8 chains x 12 windows of 40 blocks x 10 steps (L up to 400), 100 walkers,
# h = 0.04, 0.02, 0.01 A.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_fd.py $c 100 12 40 10 0.04,0.02,0.01 pilot_$c.npz > pilot_$c.log 2>&1 &
done
wait
echo "PILOT_DONE $(date)"
