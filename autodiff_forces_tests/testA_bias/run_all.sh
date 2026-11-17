#!/bin/bash
# Test A on ne: 8 independent chains (cores 16-23), 100 walkers, windows of 4 blocks x 10 steps, 1000 windows each.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
for c in 0 1 2 3 4 5 6 7; do
  taskset -c $((16 + c)) /fs/home/sajad/.venv/bin/python -u run_bias.py $c 100 1000 4 10 1e-3,1e-2,3e-2,1e-1 chain_$c.npz > chain_$c.log 2>&1 &
done
wait
echo "TESTA_DONE $(date)"
