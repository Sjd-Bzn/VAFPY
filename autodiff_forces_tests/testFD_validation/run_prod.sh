#!/bin/bash
# Production on ne (cores 16-23), started after the pilot: ensemble FD check, 8 chains x 250 windows of
# 40 blocks x 10 steps (L up to 400), h = 0.2, 0.1, 0.05 (the pilot showed smaller h to be too noisy).
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
until grep -q PATH_DONE run_pilot.log; do sleep 20; done
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_fd.py $((10 + c)) 100 250 40 10 0.2,0.1,0.05 prod_fd_$c.npz > prod_fd_$c.log 2>&1 &
done
wait
echo "PROD_FD_DONE $(date)"
