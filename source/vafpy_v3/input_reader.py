"""Reads vafpy.in YAML control file into a namespace."""
import os
import numpy as np
from scipy.linalg import block_diag
from mpi4py import MPI
import yaml
from types import SimpleNamespace


def read(path="vafpy.in"):
    afqmc = SimpleNamespace()

    with open(path, "r") as f:
        inputs = yaml.safe_load(f)

    # System
    afqmc.system = inputs.get("SYSTEM", "Autorun")
    afqmc.num_g = inputs["NGVEC"]
    afqmc.num_electrons_up = inputs["NEUP"]
    afqmc.num_electrons_down = inputs["NEDOWN"]
    afqmc.num_orb = inputs["NORB"]
    afqmc.H_zero = inputs.get("EHZB", 0)
    afqmc.en_const = inputs.get("ECONST", 0)
    afqmc.SPIN = inputs.get("SPIN", 0)
    afqmc.EBANDS = inputs.get("EBANDS", 0)
    afqmc.num_k = inputs.get("KPOINT", 1)
    afqmc.fsg = inputs.get("FSG", 0)

    # Control
    afqmc.MAX_RUN_TIME = inputs.get("TMAX", 184800)
    afqmc.EQUILIBRATION = inputs.get("EQLB", 0.1)
    afqmc.D_TAU = inputs["DTAU"]
    afqmc.SQRT_DTAU = np.sqrt(afqmc.D_TAU)
    afqmc.NUM_WALKERS = inputs.get("NWAK", 256)
    afqmc.NUM_STEPS = inputs.get("NSTP", 1000)
    afqmc.block_divisor = inputs.get("BLKDIV", 10)
    afqmc.propagator = inputs["PROPAG"]
    afqmc.order_trunc = inputs.get("OTEY", 6)
    afqmc.trsh_imag = inputs.get("TIMG", 2000)
    afqmc.trsh_real = inputs.get("TREL", 2000)

    # Method implementation
    afqmc.UPDATE_METHOD = inputs.get("UPDTM", "H")
    afqmc.REORTHO_PERIODICITY = inputs.get("REOPRI", 1)
    afqmc.REBAL_PERIODICITY = inputs.get("REBPRI", 5)
    afqmc.SAMP_FREQ = inputs.get("SMFRQ", 1)
    afqmc.CHECK_PERIODICITY = inputs.get("CHPRI", 1000)

    # SVD / HF test
    afqmc.SVD = inputs.get("SVD", False)
    afqmc.svd_trshd = inputs.get("SVDT", 1e-5)
    afqmc.HF_TEST = inputs.get("HF", False)
    if afqmc.HF_TEST:
        afqmc.HF_TEST_tau = inputs.get("HFTAU", 1.0e-10)
        afqmc.HF_TEST_steps = inputs.get("HFSTP", 1)

    # Backend / precision
    afqmc.Backend = inputs.get("BACKEND", "NumPy")
    afqmc.Precsion = inputs.get("PRECSION", "Double")

    # MPI
    afqmc.comm = MPI.COMM_WORLD
    afqmc.size = afqmc.comm.Get_size()
    afqmc.NUM_WALKERS = afqmc.NUM_WALKERS // afqmc.size
    afqmc.rank = afqmc.comm.Get_rank()
    afqmc.first_cpu = afqmc.rank == 0
    afqmc.seed = inputs.get("SEED", None)

    # Input data files (kpoint-aware naming)
    afqmc.input_file_one_body_hamil = inputs.get("H1FILE", "H1_svd.npy")
    afqmc.input_file_two_body_hamil = inputs.get("H2FILE", "H2_zip.npy")
    afqmc.q_list_file = inputs.get("QLIST", "Q_list.npy")

    # Build single-k trial and block-diagonal multi-k trial
    psi_t_single = np.eye(afqmc.num_orb)[:, : afqmc.num_electrons_up]
    afqmc.PSI_T_up_0 = psi_t_single
    afqmc.PSI_T_down_0 = psi_t_single
    afqmc.PSI_T_up = psi_t_single
    afqmc.PSI_T_down = psi_t_single
    for _ in range(1, afqmc.num_k):
        afqmc.PSI_T_up = block_diag(afqmc.PSI_T_up, psi_t_single)
        afqmc.PSI_T_down = block_diag(afqmc.PSI_T_down, psi_t_single)

    if afqmc.SPIN == 0:
        afqmc.output_file = (
            f"AFQMC_CS_num_k{afqmc.num_k}_samp_freq_{afqmc.SAMP_FREQ}"
            f"_Reortho_{afqmc.REORTHO_PERIODICITY}"
            f"_Rebal_{afqmc.REBAL_PERIODICITY}"
            f"_TAU_{afqmc.D_TAU}_walkers_{afqmc.NUM_WALKERS}.txt"
        )
    else:
        afqmc.output_file = (
            f"AFQMC_SP_num_k{afqmc.num_k}_{afqmc.system}_"
            f"{afqmc.UPDATE_METHOD}_Reortho_{afqmc.REORTHO_PERIODICITY}"
            f"_Rebal_{afqmc.REBAL_PERIODICITY}_TAU_{afqmc.D_TAU}.txt"
        )

    return afqmc
