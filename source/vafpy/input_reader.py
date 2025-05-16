import numpy as np
from scipy.linalg import block_diag
from mpi4py import MPI
import yaml 
from types import SimpleNamespace

def read():
    afqmc = SimpleNamespace()

    with open ("INCAR_AFQMC", "r") as f:
        inputs = yaml.safe_load(f)


    afqmc.system = inputs.get('SYSTEM', 'Autorun')


    afqmc.num_g =  inputs["NGVEC"]                                   ### from the out put file of the HF in H2 part

    afqmc.num_electrons_up = inputs['NEUP']

    afqmc.num_electrons_down = inputs['NEDOWN']

    afqmc.block_divisor = inputs.get('BLKDIV', 10)

    afqmc.num_orb =inputs["NORB"]                                             ##### from the H1.npy or H2.npy

    afqmc.H_zero = inputs.get("EHZB", 0) 

    afqmc.en_const =  inputs["ECONST"]                            #nuclear energy from the output of the HF

    afqmc.SPIN =inputs.get("SPIN", 0)

    afqmc.EBANDS = inputs.get("EBANDS", 0)


    afqmc.MAX_RUN_TIME =inputs.get("TMAX", 184800)

    afqmc.EQUILIBRATION = inputs.get("EQLB", 0.1)

    afqmc.D_TAU=inputs["DTAU"] ##0.005

    afqmc.SQRT_DTAU = np.sqrt(afqmc.D_TAU)

    afqmc.NUM_WALKERS = inputs.get("NWAK", 256)

    afqmc.NUM_STEPS =inputs.get("NSTP", 1000)

    afqmc.comm = MPI.COMM_WORLD
    afqmc.size = afqmc.comm.Get_size()
    afqmc.NUM_WALKERS=afqmc.NUM_WALKERS//afqmc.size
    afqmc.rank = afqmc.comm.Get_rank()
    #base_seed = 12345
    #seed = base_seed + 9999 * rank
    #np.random.seed(seed)


    afqmc.propagator = inputs["PROPAG"]


    afqmc.order_trunc =inputs.get("OTEY", 6)                                     ### exp_Taylor (matrics to ex)

    afqmc.trsh_imag =inputs.get("TIMG", 2000)                                    ### ???

    afqmc.trsh_real =inputs.get("TREL", 2000)



    afqmc.first_cpu =afqmc.comm.Get_rank()==0


    afqmc.UPDATE_METHOD =inputs.get("UPDTM", "H")

    afqmc.REORTHO_PERIODICITY =inputs.get("REOPRI", 1)

    afqmc.REBAL_PERIODICITY = inputs.get("REBPRI", 5)

    afqmc.SAMP_FREQ =inputs.get("SMFRQ", 1) 

    afqmc.CHECK_PERIODICITY =inputs.get("CHPRI", 1000)

    afqmc.Backend = inputs.get("BACKEND", "NumPy")

    afqmc.Precsion = inputs.get("PRECSION", "Double")


    afqmc.SVD =inputs.get("SVD",False)                         

    if afqmc.SVD==True:
        afqmc.svd_trshd = inputs.get("SVDT", 1e-3)

    afqmc.svd_trshd = inputs.get("SVDT", 1e-5)


    afqmc.HF_TEST = inputs.get("HF", False)                                    #### HF_TEST
    if afqmc.HF_TEST==True:
        afqmc.HF_TEST_tau =inputs.get("HFTAU", 1.0e-10)
        afqmc.HF_TEST_steps =inputs.get("HFSTP", 1)   


    afqmc.num_k =inputs.get("KPOINT", 1)                                               #### closed or open shell

    afqmc.fsg = inputs.get("FSG", 0)

    afqmc.input_file_one_body_hamil = 'H1.npy'
    afqmc.input_file_two_body_hamil = 'H2_zip.npy'
    afqmc.q_list = 'Q_list.npy'


    #if num_k==8:
    #    fsg = 8.08288908510240 
    #    input_file_one_body_hamil = 'H1.npy'
    #    input_file_two_body_hamil = 'H2.npy'
    #    h_en_vasp = 24.58880813
    #    ex_en_vasp = -87.45880075
    #    Ec_MP2_vasp = -2.61184550
    #    q_list = 'Q_list.npy'
    #
    #if num_k==1:
    #    fsg = 0.0
    #    input_file_one_body_hamil = 'H1.npy'
    #    input_file_two_body_hamil = 'H2.npy'
    #    h_en_vasp = 0.0
    #    ex_en_vasp = 0.0
    #    Ec_MP2_vasp = 0.0
    #    q_list = 'Q_list.npy'
    #    


    afqmc.PSI_T_up_0 = afqmc.PSI_T_down_0 = np.eye(afqmc.num_orb)[:,0:afqmc.num_electrons_up]

    afqmc.PSI_T_up = afqmc.PSI_T_up_0

    afqmc.PSI_T_down = afqmc.PSI_T_down_0
    for i in range (1,afqmc.num_k):
        afqmc.PSI_T_up = block_diag(afqmc.PSI_T_up,afqmc.PSI_T_up_0)
        afqmc.PSI_T_down = block_diag(afqmc.PSI_T_down,afqmc.PSI_T_down_0)


    if afqmc.SPIN==0:
        afqmc.output_file = 'AFQMC_CS_num_k'+str(afqmc.num_k)+'_samp_freq_'+str(afqmc.SAMP_FREQ)+'_Reortho_'+str(afqmc.REORTHO_PERIODICITY) +'_Rebal_'+str(afqmc.REBAL_PERIODICITY)+'_TAU_'+str(afqmc.D_TAU)+'_walkers_'+str(afqmc.NUM_WALKERS)+'.txt'
        
    elif afqmc.SPIN==1:
        afqmc.output_file = 'AFQMC_SP_num_k'+str(afqmc.num_k)+'_'+afqmc.system+'_'+str(afqmc.UPDATE_METHOD)+'_Reortho_'+str(afqmc.REORTHO_PERIODICITY) +'_Rebal_'+str(afqmc.REBAL_PERIODICITY)+'_TAU_'+str(afqmc.D_TAU)+'.txt'

    return afqmc
