import numpy as np
from scipy.linalg import block_diag
#from mpi4py import MPI

num_g = 170                                      ### from the out put file of the HF in H2 part

#H_zero = -142.09249198310414

en_const =  9.20490701                            #nuclear energy from the output of the HF

SVD = False                                         

if SVD==True:
    svd_trshd = 1e-3

svd_trshd = 1e-3

HF_TEST = False                                     #### HF_TEST

if HF_TEST==True:
    HF_TEST_tau = 1.0e-10
    HF_TEST_steps = 1

order_trunc = 6                                     ### exp_Taylor (matrics to ex)

trsh_imag = 2000                                    ### ???

trsh_real = 2000

MAX_RUN_TIME = 184800

SPIN = 0

num_electrons_up = 5

num_electrons_down = 5

num_orb = 24                                            ##### from the H1.npy or H2.npy

num_k = 1                                               #### close or open shell

PSI_T_up_0 = PSI_T_down_0 = np.eye(num_orb)[:,0:num_electrons_up]

PSI_T_up = PSI_T_up_0

PSI_T_down = PSI_T_down_0

for i in range (1,num_k):
    PSI_T_up = block_diag(PSI_T_up,PSI_T_up_0)
    PSI_T_down = block_diag(PSI_T_down,PSI_T_down_0)

D_TAU = 0.005

SQRT_DTAU = np.sqrt(D_TAU)

NUM_WALKERS = 256

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#NUM_WALKERS=NUM_WALKERS//size
first_cpu = True #comm.Get_rank()==0

NUM_STEPS = 10000

UPDATE_METHOD = "H"

REORTHO_PERIODICITY = 1

REBAL_PERIODICITY = 10

EQUILIBRATION = 0.1

SAMP_FREQ = 1

CHECK_PERIODICITY = 1000

if num_k==8:
    fsg = 8.08288908510240 
    input_file_one_body_hamil = 'H1.npy'
    input_file_two_body_hamil = 'H2.npy'
    h_en_vasp = 24.58880813
    ex_en_vasp = -87.45880075
    Ec_MP2_vasp = -2.61184550
    q_list = 'Q_list.npy'

if num_k==1:
    fsg = 0.0
    input_file_one_body_hamil = 'H1.npy'
    input_file_two_body_hamil = 'H2.npy'
    h_en_vasp = 0.0
    ex_en_vasp = 0.0
    Ec_MP2_vasp = 0.0
    q_list = 'Q_list.npy'

system = 'H2O'

if SPIN==0:
    output_file = 'AFQMC_CS_num_k'+str(num_k)+'_'+system+'_'+str(UPDATE_METHOD)+'_Reortho_'+str(REORTHO_PERIODICITY) +'_Rebal_'+str(REBAL_PERIODICITY)+'_TAU_'+str(D_TAU)+'.txt'
    
elif SPIN==1:
    output_file = 'AFQMC_SP_num_k'+str(num_k)+'_'+system+'_'+str(UPDATE_METHOD)+'_Reortho_'+str(REORTHO_PERIODICITY) +'_Rebal_'+str(REBAL_PERIODICITY)+'_TAU_'+str(D_TAU)+'.txt'


