#!/bin/bash
# Test F on he (RTX A6000): the same configurations as the VSC5 CPU sweep (configs_cpu.txt), one per process, so the
# peak device memory belongs to that configuration only. Results as JSON lines in results.jsonl.
# MPI library for mpi4py from oneAPI (as in the molecule runs); pytest is not installed, minimal_pytest stands in.
source /fs/home/tobias/bin/oneapi-202501/setvars.sh >/dev/null 2>&1
source ~/.venv/vafpy_gpu/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")"
export PYTHONPATH=$PWD/minimal_pytest
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "$(date) GPU busy" >> driver.log; exit 1; }
echo "$(date) start" >> driver.log
while read -r cfg; do
  taskset -c 0-5 python -u run_mem.py $cfg >> results.jsonl 2>> errors.log || echo "{\"failed\": \"$cfg\"}" >> results.jsonl
done < configs_cpu.txt
echo "$(date) TESTF_DONE $(grep -c failed results.jsonl) failed" >> driver.log
