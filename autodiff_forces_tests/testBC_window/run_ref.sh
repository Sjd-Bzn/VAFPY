#!/bin/bash
# Test C reference on ne (cores 16-23): AFQMC energy at lam = -0.05, 0, +0.05 (V = H1), 4 independent seeds each,
# 100 walkers, 2000 blocks of 10 steps (beta = 5 eV^-1), population control every 10 steps.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
i=0
for lam in -0.05 0.0 0.05; do
  for s in 1 2 3 4; do
    while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 2; done
    taskset -c $((16 + i % 8)) /fs/home/sajad/.venv/bin/python -u run_lambda.py $((10 * s + i)) $lam 100 2000 10 ref_lam${lam}_s$s.npz > ref_lam${lam}_s$s.log 2>&1 &
    i=$((i + 1))
  done
done
wait
echo "REF_DONE $(date)"
