"""Reads the vafpy.in YAML control file into a namespace (closed-shell: NEUP electrons per spin; unknown keys are ignored)."""
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
    afqmc.num_orb = inputs["NORB"]
    afqmc.num_k = inputs.get("KPOINT", 1)
    afqmc.fsg = inputs.get("FSG", 0)

    # Control
    afqmc.EQUILIBRATION = inputs.get("EQLB", 0.1)
    afqmc.D_TAU = inputs["DTAU"]
    afqmc.NUM_WALKERS = inputs.get("NWAK", 256)
    afqmc.NUM_STEPS = inputs.get("NSTP", 1000)
    afqmc.propagator = inputs["PROPAG"]
    afqmc.order_trunc = inputs.get("OTEY", 6)

    # Method implementation
    afqmc.REORTHO_PERIODICITY = inputs.get("REOPRI", 1)
    afqmc.REORTHO_METHOD = inputs.get("REORTHO", "Auto")   # QR, Cholesky or Auto (Cholesky on JAX, QR otherwise)
    afqmc.REBAL_PERIODICITY = inputs.get("REBPRI", 5)
    afqmc.REBAL_METHOD = inputs.get("REBMTH", "local")   # local or global
    afqmc.SAMP_FREQ = inputs.get("SMFRQ", 1)
    afqmc.CHECK_PERIODICITY = inputs.get("CHPRI", 1000)

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
    afqmc.timing = int(inputs.get("TIMING", 1))   # timing report timing.txt: 0 off, 1 parts, 2 parts and stages (timing.py)

    # Input data files (kpoint-aware naming)
    afqmc.input_file_one_body_hamil = inputs.get("H1FILE", "H1_svd.npy")
    afqmc.input_file_two_body_hamil = inputs.get("H2FILE", "H2_zip.npy")
    afqmc.q_list_file = inputs.get("QLIST", "Q_list.npy")
    afqmc.q_sizes_file = inputs.get("QSIZES", "Q_sizes.npy")   # columns per q sector (compact H2)

    afqmc.output_file = (
        f"AFQMC_CS_num_k{afqmc.num_k}_samp_freq_{afqmc.SAMP_FREQ}"
        f"_Reortho_{afqmc.REORTHO_PERIODICITY}"
        f"_Rebal_{afqmc.REBAL_PERIODICITY}"
        f"_TAU_{afqmc.D_TAU}_walkers_{afqmc.NUM_WALKERS}.txt"
    )

    return afqmc
