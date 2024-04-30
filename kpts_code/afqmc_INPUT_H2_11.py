import numpy as np
from scipy.linalg import block_diag
#from mpi4py import MPI

h=np.load('H2_zip_11.npy')


num_g =  h.shape[2] #4291

del(h)

en_const = 0.15490133-0.61389897-47.85422692+47.83088869+18.91110697

SVD = False

if SVD==True:
    svd_trshd = 1e-12

svd_trshd = 1e-3

HF_TEST = False

if HF_TEST==True:
    HF_TEST_tau = 1.0e-10
    HF_TEST_steps = 1

order_trunc = 6

trsh_imag = 2000

trsh_real = 2000

MAX_RUN_TIME = 184800

SPIN = 0

num_electrons_up = 4

num_electrons_down = 4

num_orb = 8

num_k = 8

PSI_T_up_0 = PSI_T_down_0 = np.eye(num_orb)[:,0:num_electrons_up]

PSI_T_up = PSI_T_up_0

PSI_T_down = PSI_T_down_0

for i in range (1,num_k):
    PSI_T_up = block_diag(PSI_T_up,PSI_T_up_0)
    PSI_T_down = block_diag(PSI_T_down,PSI_T_down_0)

D_TAU = 0.005

SQRT_DTAU = np.sqrt(D_TAU)

NUM_WALKERS = 512

#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#NUM_WALKERS=NUM_WALKERS//size
first_cpu = True #comm.Get_rank()==0

NUM_STEPS = 6000

UPDATE_METHOD = "H"

REORTHO_PERIODICITY = 1

REBAL_PERIODICITY = 10

EQUILIBRATION = 0.2

SAMP_FREQ = 1

CHECK_PERIODICITY = 1499

if num_k==8:
    fsg = 8.47945518200827 
    input_file_one_body_hamil = 'H1_11.npy'
    input_file_two_body_hamil = 'H2_zip_11.npy'
    h_en_vasp = 406.35071511
    ex_en_vasp = -151.20984269
    Ec_MP2_vasp = -0.18206868
    q_list = 'Q_list.npy'

system = 'C'

if SPIN==0:
    output_file = '11_AFQMC_CS_num_k'+str(num_k)+'_'+system+'_'+str(UPDATE_METHOD)+'_Reortho_'+str(REORTHO_PERIODICITY) +'_Rebal_'+str(REBAL_PERIODICITY)+'_TAU_'+str(D_TAU)+'.txt'
    
elif SPIN==1:
    output_file = 'AFQMC_SP_num_k'+str(num_k)+'_'+system+'_'+str(UPDATE_METHOD)+'_Reortho_'+str(REORTHO_PERIODICITY) +'_Rebal_'+str(REBAL_PERIODICITY)+'_TAU_'+str(D_TAU)+'.txt'


