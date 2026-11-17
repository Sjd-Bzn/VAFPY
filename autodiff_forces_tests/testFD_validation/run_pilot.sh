#!/bin/bash
# Pilot on ne (cores 16-23): ensemble FD check, 8 chains x 12 windows of 40 blocks x 10 steps (L up to 400),
# h = 0.2, 0.1, 0.05, 0.025, 0.0125; then the pathwise check, 8 chains x 5 windows.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_fd.py $c 100 12 40 10 0.2,0.1,0.05,0.025,0.0125 pilot_fd_$c.npz > pilot_fd_$c.log 2>&1 &
done
wait
echo "PILOT_FD_DONE $(date)"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_path.py $c 100 5 40 10 1e-2,1e-3,1e-4,1e-5,1e-6 path_$c.npz > path_$c.log 2>&1 &
done
wait
echo "PATH_DONE $(date)"
