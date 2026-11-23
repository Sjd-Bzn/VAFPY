#!/bin/bash
# Independent check of dE/dlam without any derivative: normal AFQMC energy (energy only, normal comb) with H1 scaled by
# (1 + lam), lam = 0, +-0.1, +-0.2, +-0.3, 4 independent seeds each, 100 walkers, 10000 blocks of 10 steps; ne cores 16-23.
# The slope of E(lam) at lam = 0 is the one-body part of the AFQMC energy (fit in analyse_lambda_scan.py).
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=${1:-/data/sajad/calcs/autodiff_forces/testBC_lambda_scan}
PART=${2:-h1}      # h1: H1 scaled; h2: L -> (1 + lam/2) L
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT"
i=0
for lam in -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3; do
  for s in 1 2 3 4; do
    while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 5; done
    taskset -c $((16 + i % 8)) /fs/home/sajad/.venv/bin/python -u "$SRC/run_lambda.py" $((1000 + 10 * s + i)) $lam 100 10000 10 \
      scan_lam${lam}_s$s.npz $PART > scan_lam${lam}_s$s.log 2>&1 &
    i=$((i + 1))
  done
done
wait
echo "SCAN_DONE $(date)"
