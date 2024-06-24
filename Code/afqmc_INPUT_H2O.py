import numpy as np
from scipy.linalg import block_diag
#from mpi4py import MPI
import yaml 
with open ("INCAR_AFQMC", "r") as f:
    inputs = yaml.safe_load(f)

system = inputs.get('SYSTEM', 'Autorun')


num_g =     inputs["NGVEC"]                                   ### from the out put file of the HF in H2 part

num_electrons_up = 5

num_electrons_down = 5

num_orb =inputs["NORB"]                                             ##### from the H1.npy or H2.npy

H_zero = inputs.get("EHZB", 0) 

en_const =  inputs["ECONST"]                            #nuclear energy from the output of the HF

SPIN =inputs.get("SPIN", 0)



MAX_RUN_TIME =inputs.get("TMAX", 184800)

EQUILIBRATION = inputs.get("EQLB", 0.1)

D_TAU=inputs["DTAU"] ##0.005

SQRT_DTAU = np.sqrt(D_TAU)

NUM_WALKERS = inputs.get("NWAK", 256)

NUM_STEPS =inputs.get("NSTP", 1000)

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#NUM_WALKERS=NUM_WALKERS//size

order_trunc =inputs.get("OTEY", 6)                                     ### exp_Taylor (matrics to ex)

trsh_imag =inputs.get("TIMG", 2000)                                    ### ???

trsh_real =inputs.get("TREL", 2000)



first_cpu = True #comm.Get_rank()==0


UPDATE_METHOD =inputs.get("UPDTM", "H")

REORTHO_PERIODICITY =inputs.get("REOPRI", 1)

REBAL_PERIODICITY = inputs.get("REBPRI", 5)

SAMP_FREQ =inputs.get("SMFRQ", 1) 

CHECK_PERIODICITY =inputs.get("CHPRI", 1000)



SVD =inputs.get("SVD",False)                         

if SVD==True:
    svd_trshd = inputs.get("SVDT", 1e-3)

svd_trshd = inputs.get("SVDT", 1e-3)


HF_TEST = inputs.get("HF", False)                                    #### HF_TEST
if HF_TEST==True:
    HF_TEST_tau =inputs.get("HFTAU", 1.0e-10)
    HF_TEST_steps =inputs.get("HFSTP", 1)   


num_k =inputs.get("KPOINT", 1)                                               #### closed or open shell

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
    


PSI_T_up_0 = PSI_T_down_0 = np.eye(num_orb)[:,0:num_electrons_up]

PSI_T_up = PSI_T_up_0

PSI_T_down = PSI_T_down_0

for i in range (1,num_k):
    PSI_T_up = block_diag(PSI_T_up,PSI_T_up_0)
    PSI_T_down = block_diag(PSI_T_down,PSI_T_down_0)



if SPIN==0:
    output_file = 'AFQMC_CS_num_k'+str(num_k)+'_'+system+'_'+str(UPDATE_METHOD)+'_Reortho_'+str(REORTHO_PERIODICITY) +'_Rebal_'+str(REBAL_PERIODICITY)+'_TAU_'+str(D_TAU)+'.txt'
    
elif SPIN==1:
    output_file = 'AFQMC_SP_num_k'+str(num_k)+'_'+system+'_'+str(UPDATE_METHOD)+'_Reortho_'+str(REORTHO_PERIODICITY) +'_Rebal_'+str(REBAL_PERIODICITY)+'_TAU_'+str(D_TAU)+'.txt'


