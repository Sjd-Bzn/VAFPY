"""Cross-backend smoke test: NumPy and JAX produce matching HF energies."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import functions as new


def run_hf(backend_name):
    backend = new.make_backend(backend_name, seed=12345)
    from mpi4py import MPI
    config = new.Configuration(
        num_walkers=10, num_kpoint=1, num_orbital=8, num_electron=4,
        num_g=36, singularity=0.0, propagator="S2", order_propagation=6,
        timestep=0.005, comm=MPI.COMM_WORLD, precision="Double",
        backend=backend,
    )
    ql = new.obtain_Q_list(config, "Q_list.npy")
    h1 = new.obtain_H1(config, "H1_svd.npy")
    h2 = new.obtain_H2(config, "H2_zip.npy")
    H = new.Hamiltonian(one_body=h1, two_body=h2, q_list=ql)
    trial, walkers = new.initialize_determinant(config)
    H.setup_energy_expressions(config, trial)
    e_total, _, _ = new.measure_energy(config, trial, walkers, H)
    return complex(e_total), complex(H.H_zero)


def main():
    e_np, h0_np = run_hf("numpy")
    print(f"NumPy backend: HF = {e_np}, H_zero = {h0_np}")
    e_jax, h0_jax = run_hf("jax")
    print(f"JAX   backend: HF = {e_jax}, H_zero = {h0_jax}")
    # JAX defaults to single precision; set JAX_ENABLE_X64=1 for fp64 parity.
    tol = 1e-4
    assert abs(e_np - e_jax) < tol, (e_np, e_jax)
    assert abs(h0_np - h0_jax) < tol, (h0_np, h0_jax)
    print(f"[PASS] NumPy and JAX backends agree to {tol} "
          "(JAX runs in fp32 unless JAX_ENABLE_X64=1 is set).")


if __name__ == "__main__":
    main()
