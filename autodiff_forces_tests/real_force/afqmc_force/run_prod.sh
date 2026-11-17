#!/bin/bash
# Production on ne (cores 16-23): 8 chains x 300 windows of 40 blocks x 10 steps (L up to 400), 100 walkers,
# h = 0.16, 0.08, 0.04 A (the pilot showed the FD spread to grow for smaller h).
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_fd.py $((10 + c)) 100 300 40 10 0.16,0.08,0.04 prod_$c.npz > prod_$c.log 2>&1 &
done
wait
echo "PROD_DONE $(date)"
