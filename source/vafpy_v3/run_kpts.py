"""AFQMC driver for vafpy_v2 (GPU/CPU + k-point support)."""
import numpy as np
from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() != 0:
    print = lambda *a, **kw: None  # noqa: E731

from time import time
from os.path import exists

import input_reader
import functions as new


def main():
    afqmc = input_reader.read()

    if afqmc.first_cpu:
        print("###########################")
        print("    AFQMC (vafpy_v2)")
        print("###########################")
        print()
        print("system               = ", afqmc.system)
        print("num_k                = ", afqmc.num_k)
        print("num_orb              = ", afqmc.num_orb)
        print("num_electrons_up     = ", afqmc.num_electrons_up)
        print("num_g                = ", afqmc.num_g)
        print("D_TAU                = ", afqmc.D_TAU)
        print("NUM_WALKERS / rank   = ", afqmc.NUM_WALKERS)
        print("NUM_STEPS            = ", afqmc.NUM_STEPS)
        print("propagator           = ", afqmc.propagator)
        print("backend              = ", afqmc.Backend)
        print("precision            = ", afqmc.Precsion)
        print("REORTHO_PERIODICITY  = ", afqmc.REORTHO_PERIODICITY)
        print("REBAL_PERIODICITY    = ", afqmc.REBAL_PERIODICITY)
        print("output_file          = ", afqmc.output_file)
        print()

    # Pick seed: input file > random
    rank = afqmc.comm.Get_rank()
    if afqmc.seed is None:
        max_seed = np.iinfo(np.int32).max
        seed = np.random.randint(max_seed)
    else:
        seed = int(afqmc.seed) + 9999 * rank
    np.random.seed(seed)

    backend = new.make_backend(afqmc.Backend, seed)

    config = new.Configuration(
        num_walkers=afqmc.NUM_WALKERS,
        num_kpoint=afqmc.num_k,
        num_orbital=afqmc.num_orb,
        num_electron=afqmc.num_electrons_up,
        num_g=afqmc.num_g,
        singularity=afqmc.fsg,
        propagator=afqmc.propagator,
        order_propagation=afqmc.order_trunc,
        timestep=afqmc.D_TAU,
        comm=afqmc.comm,
        precision=afqmc.Precsion,
        backend=backend,
    )

    q_list = new.obtain_Q_list(config, afqmc.q_list_file)
    hamiltonian = new.Hamiltonian(
        one_body=new.obtain_H1(config, afqmc.input_file_one_body_hamil),
        two_body=new.obtain_H2(
            config, afqmc.input_file_two_body_hamil, q_list=q_list
        ),
        q_list=q_list,
    )
    trial_det, walkers = new.initialize_determinant(config)
    hamiltonian.setup_energy_expressions(config, trial_det)

    if afqmc.first_cpu:
        print("Hamiltonian set up. H_zero (mean-field constant) =",
              hamiltonian.H_zero)
        print()

    e1, eh, ex = new.measure_components(config, trial_det, walkers, hamiltonian)
    a = new.measure_energy(config, trial_det, walkers, hamiltonian)
    energy_new = e_hf = a[0]
    if afqmc.first_cpu:
        print("Initial HF components:")
        print("  E_one    =", e1)
        print("  Hartree  =", eh)
        print("  Exchange =", ex)
        print("  E_total  =", e_hf)
        print()

    # Per-walker constant factor used by S2 propagator.
    h_0 = np.exp(config.timestep * hamiltonian.H_zero)

    # Open output files
    wr_name = "WALKERS.npy"
    wt_name = "WEIGHT.npy"
    file_exists = exists(wr_name)
    mode = "a" if file_exists else "w"
    file_out = open(afqmc.output_file, mode)
    weights_file = open("weights_history.txt", "w")
    weights_file.write(f"{0}:   {np.mean(config.backend.to_numpy(walkers.weights))}\n")

    rare_event_total = 0
    rare_event_steps = 0
    start_time = time()
    NUM_WALKERS = afqmc.NUM_WALKERS

    j = 1
    while j < afqmc.NUM_STEPS + 1:
        new_walkers, num_rare = new.propagate_walkers(
            config, trial_det, walkers, hamiltonian, h_0, energy_new
        )
        walkers = new_walkers
        rare_event_total += num_rare
        if num_rare > 0:
            rare_event_steps += 1

        avg_w = np.mean(config.backend.to_numpy(walkers.weights))
        weights_file.write(f"{j}:   {avg_w}\n")

        if j % afqmc.CHECK_PERIODICITY == 0:
            np.save(wr_name, config.backend.to_numpy(walkers.slater_det))
            np.save(wt_name, config.backend.to_numpy(walkers.weights))

        if j % afqmc.SAMP_FREQ == 0:
            a = new.measure_energy(config, trial_det, walkers, hamiltonian)
            energy_new = a[0]
            b = a[1]
            c = a[2]
            e_hf = (e_hf * j + energy_new) / (j + 1)
            if afqmc.first_cpu:
                print(j * afqmc.D_TAU, e_hf)
                file_out.write(
                    f"{j * afqmc.D_TAU}\t{energy_new.real}\t{energy_new.imag}"
                    f"\t{b.real}\t{c.real}\n"
                )
                if j % 5000 == 0:
                    file_out.flush()

        if j % afqmc.REORTHO_PERIODICITY == 0:
            walkers.slater_det = new.reortho_qr(config, walkers.slater_det)

        if afqmc.REBAL_PERIODICITY != 0 and j % afqmc.REBAL_PERIODICITY == 0:
            idx = new.rebalance_comb(config, walkers.weights)
            walkers.slater_det = walkers.slater_det[idx]
            walkers.weights = new.init_walkers_weights(config, NUM_WALKERS)

        j += 1

    file_out.close()
    weights_file.close()

    if afqmc.first_cpu:
        print("Total rare events =", rare_event_total)
        print("Total runtime     =", -start_time + time())
        print("Output file       =", afqmc.output_file)
        print()

    with open(afqmc.output_file, "r") as f:
        data = np.loadtxt(f)
    if data.ndim == 1:
        data = data[None, :]
    col = data[:, 1]
    st = int(len(col) * afqmc.EQUILIBRATION)
    en = len(col)
    v, blockVar, blockMean = new.blockAverage(col[st:en], afqmc.block_divisor)
    if afqmc.first_cpu:
        print("E_gs_afqmc =", np.mean(col[st:en]),
              "+-", np.sqrt(max(blockVar)) if len(blockVar) else float("nan"))
        with open("outcar.txt", "a") as outcar:
            outcar.write("Calculating ground state energy\n")
            outcar.write(f"E_gs_afqmc = {np.mean(col[st:en])}\n")
            outcar.write(f"+- {np.sqrt(max(blockVar)) if len(blockVar) else float('nan')}\n")
            outcar.write(f"Number of steps with rare events = {rare_event_steps}\n")
            outcar.write(f"Total number of rare events = {rare_event_total}\n")


if __name__ == "__main__":
    main()
