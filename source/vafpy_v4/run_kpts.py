"""AFQMC driver for vafpy_v4 (GPU/CPU, k-points, closed shell or spin-unrestricted)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() != 0:
    print = lambda *a, **kw: None  # noqa: E731

from time import time
from os.path import exists

import input_reader
import functions as new
import stat_analysis


def main():
    afqmc = input_reader.read()

    if afqmc.first_cpu:
        print("###########################")
        print("    AFQMC (vafpy_v4)")
        print("###########################")
        print()
        print("system               = ", afqmc.system)
        print("num_k                = ", afqmc.num_k)
        print("num_orb              = ", afqmc.num_orb)
        print("spin                 = ", "unrestricted" if afqmc.SPIN else "closed shell")
        print("num_electrons_up     = ", afqmc.num_electrons_up)
        if afqmc.SPIN:
            print("num_electrons_down   = ", afqmc.num_electrons_down)
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
        spin_polarized=bool(afqmc.SPIN),
        num_electron_dn=afqmc.num_electrons_down,
    )

    q_list = new.obtain_Q_list(config, afqmc.q_list_file)
    h2_host, layout = new.obtain_H2_host(config, afqmc.input_file_two_body_hamil)
    q_sizes = new.obtain_Q_sizes(config, afqmc.q_sizes_file) if layout == "compact" else None
    if afqmc.first_cpu:
        print("H2 layout            = ", layout, h2_host.shape)
    h1_dn = h2_host_dn = None
    if len(config.channels) == 2:
        h2_host_dn, layout_dn = new.obtain_H2_host(config, afqmc.input_file_two_body_hamil_down)
        if layout_dn != layout or h2_host_dn.shape != h2_host.shape:
            raise ValueError(f"spin-down H2 {h2_host_dn.shape} ({layout_dn}) does not match spin up "
                             f"{h2_host.shape} ({layout})")
        h1_dn = new.obtain_H1(config, afqmc.input_file_one_body_hamil_down)
        if afqmc.first_cpu:
            print("H1/H2 spin down      = ", afqmc.input_file_one_body_hamil_down, afqmc.input_file_two_body_hamil_down)
    hamiltonian = new.build_hamiltonian(
        config, new.obtain_H1(config, afqmc.input_file_one_body_hamil),
        h2_host, layout, q_list, q_sizes, one_body_dn=h1_dn, h2_host_dn=h2_host_dn)
    del h2_host, h2_host_dn
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
    h_0 = config.complex_type(np.exp(config.timestep * hamiltonian.H_zero))

    # Open output files
    wr_name = "WALKERS.npy"
    wt_name = "WEIGHT.npy"
    file_exists = exists(wr_name)
    mode = "a" if file_exists else "w"
    # only the first rank writes the output files (the other ranks never touch them)
    file_out = open(afqmc.output_file, mode) if afqmc.first_cpu else None
    weights_file = open("weights_history.txt", "w") if afqmc.first_cpu else None
    avg_w0 = np.mean(config.backend.to_numpy(walkers.weights))
    if afqmc.first_cpu:
        weights_file.write(f"{0}:   {avg_w0}\n")

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

        if j % afqmc.SAMP_FREQ == 0 and afqmc.first_cpu:       # one device-to-host copy per sample, not per step
            weights_file.write(f"{j}:   {np.mean(config.backend.to_numpy(walkers.weights))}\n")

        if j % afqmc.CHECK_PERIODICITY == 0:
            np.save(wr_name, config.backend.to_numpy(walkers.slater_det))
            if walkers.slater_det_dn is not None:
                np.save("WALKERS_down.npy", config.backend.to_numpy(walkers.slater_det_dn))
            np.save(wt_name, config.backend.to_numpy(walkers.weights))

        if j % afqmc.SAMP_FREQ == 0:
            a = new.measure_energy(config, trial_det, walkers, hamiltonian)
            energy_new = a[0]
            if not np.isfinite(energy_new):
                if afqmc.first_cpu:
                    file_out.flush()
                    weights_file.flush()
                raise FloatingPointError(
                    f"non-finite local energy at step {j} (tau*step = {j * afqmc.D_TAU})")
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
            new.reortho_walkers(config, walkers, afqmc.REORTHO_METHOD)

        if afqmc.REBAL_PERIODICITY != 0 and j % afqmc.REBAL_PERIODICITY == 0:
            if afqmc.REBAL_METHOD == "global":
                # the spin channels travel together: their columns are joined for the exchange between ranks
                dets_np = [config.backend.to_numpy(d) for d in walkers.dets]
                split = np.cumsum([d.shape[-1] for d in dets_np])[:-1]
                mats_np = dets_np[0] if len(dets_np) == 1 else np.concatenate(dets_np, axis=-1)
                wts_np  = config.backend.to_numpy(walkers.weights)
                mats_np, wts_np = new.rebalance_global(
                    config.comm, mats_np, wts_np, config)
                walkers.dets = [config.backend.array(m, dtype=config.complex_type)
                                for m in np.split(mats_np, split, axis=-1)]
                walkers.weights    = config.backend.array(wts_np,  dtype=config.complex_type)
            else:
                idx = new.rebalance_comb(config, walkers.weights)
                walkers.dets = [d[idx] for d in walkers.dets]
                walkers.weights = new.init_walkers_weights(config, NUM_WALKERS)

        j += 1

    if afqmc.first_cpu:
        file_out.close()
        weights_file.close()
        print("Total rare events =", rare_event_total)
        print("Total runtime     =", -start_time + time())
        print("Output file       =", afqmc.output_file)
        print()

    # the statistics are evaluated by the first rank from the closed file; the others wait for it
    config.comm.Barrier()
    if not afqmc.first_cpu:
        return

    with open(afqmc.output_file, "r") as f:
        data = np.loadtxt(f)
    if data.ndim == 1:
        data = data[None, :]
    col = data[:, 1]
    st = int(len(col) * afqmc.EQUILIBRATION)
    en = len(col)
    # standard error of the mean from the integrated autocorrelation time
    std_err, tau_int, tau_window_closed = new.autocorr_error(col[st:en])
    if afqmc.first_cpu:
        print("E_gs_afqmc =", np.mean(col[st:en]), "+-", std_err)
        # variance of the mean = 2 tau_int var / N; correlation length = 2 tau_int (samples per independent sample)
        var_series = float(np.var(col[st:en], ddof=1))
        var_of_mean = std_err**2
        correlation_length = 2.0 * tau_int
        print("Variance of the energy series =", var_series)
        print("Variance of the mean =", var_of_mean)
        print("Integrated autocorrelation time (samples) =", tau_int)
        print("Correlation length =", correlation_length)
        if not tau_window_closed:
            print("WARNING: the autocorrelation window did not close; the series is too short for a reliable error.")
        # statistics report: equilibration region, blocking curve, autocorrelation (figure only if matplotlib is available)
        stat_report = None
        try:
            stat_report = stat_analysis.analyse(data[:, 0], col, afqmc.EQUILIBRATION)
            stat_analysis.write_figure(stat_report, "stat_analysis.png", title=afqmc.output_file)
        except Exception as exc:
            print("WARNING: statistics report failed:", exc)
        if stat_report is not None:
            for line in stat_analysis.summary_lines(stat_report):
                print(line)
        with open("outcar.txt", "a") as outcar:
            outcar.write("Calculating ground state energy\n")
            outcar.write(f"E_gs_afqmc = {np.mean(col[st:en])}\n")
            outcar.write(f"+- {std_err}\n")
            outcar.write(f"Variance of the energy series = {var_series}\n")
            outcar.write(f"Variance of the mean = {var_of_mean}\n")
            outcar.write(f"Integrated autocorrelation time (samples) = {tau_int}\n")
            outcar.write(f"Correlation length = {correlation_length}\n")
            if stat_report is not None:
                for line in stat_analysis.summary_lines(stat_report):
                    outcar.write(line + "\n")
            outcar.write(f"Number of steps with rare events = {rare_event_steps}\n")
            outcar.write(f"Total number of rare events = {rare_event_total}\n")


def run():
    """Run main(); if any rank fails, stop the whole MPI job instead of leaving the others waiting."""
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    run()
