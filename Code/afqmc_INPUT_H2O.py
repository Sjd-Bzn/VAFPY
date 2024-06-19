import numpy as np
from scipy.linalg import block_diag
#from mpi4py import MPI
import yaml 
with open("AFQMC_INCAR", "r") as f:
    inputs = yaml.safe_load(f)
    print(inputs)

num_g =inputs.get("NG", 170)                                   ### from the out put file of the HF in H2 part

H_zero = inputs.get("H_zero", 0) 

en_const =  inputs["en_const"]                            #nuclear energy from the output of the HF

SVD =inputs.get("SVD",False)                                         

if SVD==True:
    svd_trshd = inputs.get("svd_trshd", 1e-3)

svd_trshd = inputs.get("svd_trshd", 1e-3)

HF_TEST = inputs.get("HF_TEST", False)                                    #### HF_TEST
print(num_g, HF_TEST)
exit()
if HF_TEST==True:
    HF_TEST_tau =inputs.get("HF_TEST_tau", 1.0e-10)
    HF_TEST_steps =inputs.get("HF_TEST_steps", 1)

order_trunc =inputs.get("oreder_trunc", 6)                                     ### exp_Taylor (matrics to ex)

trsh_imag =inputs.get("trsh_imag", 2000)                                    ### ???

trsh_real =inputs.get("trsh_real", 2000)

MAX_RUN_TIME =inputs.get("MAX_RUN_TIME", 184800)

SPIN =inputs.get("SPIN", 0)

num_electrons_up = 5

num_electrons_down = 5

num_orb =inputs["num_orb"]                                             ##### from the H1.npy or H2.npy

num_k =inputs.get("num_k", 1)                                               #### closed or open shell

PSI_T_up_0 = PSI_T_down_0 = np.eye(num_orb)[:,0:num_electrons_up]

PSI_T_up = PSI_T_up_0

PSI_T_down = PSI_T_down_0

for i in range (1,num_k):
    PSI_T_up = block_diag(PSI_T_up,PSI_T_up_0)
    PSI_T_down = block_diag(PSI_T_down,PSI_T_down_0)

D_TAU=inputs["D_TAU"] ##0.005

SQRT_DTAU = np.sqrt(D_TAU)

NUM_WALKERS = inputs.get("NUM_WALKERS", 256)

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#NUM_WALKERS=NUM_WALKERS//size
first_cpu = True #comm.Get_rank()==0

NUM_STEPS =inputs.get("NUM_STEPS", 1000)

UPDATE_METHOD =inputs.get("UPDATE_METHOD", "H")

REORTHO_PERIODICITY =inputs.get("REORTHO_PERIODICITY", 1)

REBAL_PERIODICITY = inputs.get("REBAL_PERIODICITY", 5)

EQUILIBRATION = inputs.get("EQUILIBRATION", 0.1)

SAMP_FREQ =inputs.get("SAMP_FREQ", 1) 

CHECK_PERIODICITY =inputs.get("CHECK_PERIODICITY", 1000)

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

system = inputs.get('SYSTEM', 'SYSTEM_NAME')

if SPIN==0:
    output_file = 'AFQMC_CS_num_k'+str(num_k)+'_'+system+'_'+str(UPDATE_METHOD)+'_Reortho_'+str(REORTHO_PERIODICITY) +'_Rebal_'+str(REBAL_PERIODICITY)+'_TAU_'+str(D_TAU)+'.txt'
    
elif SPIN==1:
    output_file = 'AFQMC_SP_num_k'+str(num_k)+'_'+system+'_'+str(UPDATE_METHOD)+'_Reortho_'+str(REORTHO_PERIODICITY) +'_Rebal_'+str(REBAL_PERIODICITY)+'_TAU_'+str(D_TAU)+'.txt'


