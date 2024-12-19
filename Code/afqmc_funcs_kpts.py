from mpi4py import MPI
import numpy as np
from time import time
from dataclasses import dataclass
from cmath import phase
from afqmc_INPUT_H2O import *
from opt_einsum import contract
from opt_einsum import contract_expression
from scipy.linalg import expm
import hashlib
#from afqmc_alpha import expr_exch_3
#from afqmc_alpha import expr_hart
from afqmc_alpha import expr_h2_e
from afqmc_alpha import expr_h2_o
#from afqmc_alpha import ALPHA_E
#from afqmc_alpha import ALPHA_O
#from afqmc_alpha import expr_exch_new
#from afqmc_alpha import expr_hart_new
from afqmc_alpha import expr_hart_new_new
from afqmc_alpha import expr_exch_new_new
from afqmc_alpha import expr_fb_e
from afqmc_alpha import expr_fb_o

@dataclass
class HAMILTONIAN:
    one_body: np.complex128   ## H_1 = sum(k) sum(pq) h(pq)(k) * a(dagger(p)(k)) * a(q)(k)  ## H_1 = np.sum((np.sum(t(pq)* np.matmul(adagger, a)))  ##adagger = np.conj(a).T 
    two_body: np.complex128  ## H_2 = 1/2 sum(q, G) L(qG) * L(dagger)(qG)   ## L(qG) = np.sqrt(4*np.pi)/np.abs(G-q) * np.sum(np.sum( rho(prkr(q, G)) * np.matmul(np.conj(a).T(pk_r +q), a(rk_r))

@dataclass
class HAMILTONIAN_MF:
    zero_body: np.complex128   ## Nuclei repulsion  H_0 = np.sum (Z(a). Z(b)/np.abs(R(a) - R(b))) 
    one_body: np.complex128    ## Kinetic and coulomb attarction  H_1 = (-1/2) np.sum(np.gradient(np.gradient(psi, r), r) H_1 = -np.sum(np.sum(Z(a)/np.abs((r(i) - R(a)))))  
    two_body_e: np.complex128  ## electron repulsion      H_2 = np.sum(1/np.abs(r(i)- r(j)))
    two_body_o: np.complex128  ## ...

@dataclass
class WALKERS:
    mats_up = np.array(NUM_WALKERS * [PSI_T_up], dtype=np.complex128)   ### spinn up and down
    mats_down = np.array(NUM_WALKERS * [PSI_T_down], dtype=np.complex128)
    weights = np.ones(NUM_WALKERS, dtype=np.complex128)   ## initiate by PSI_I from DFT calculation which at first has weight = 1 and phase = 0  

def show_params():
    print('system = ', system)
    if SVD==True:
        print('svd_trshd = ', svd_trshd)
    print('order_trunc = ', order_trunc)
    print('trsh_imag = ', trsh_imag)
    print('trsh_real = ',trsh_real)
    print('MAX_RUN_TIME = ', MAX_RUN_TIME)
    print('SPIN = ', SPIN)
    print('input_file_one_body_hamil = ', input_file_one_body_hamil)
    print('input_file_two_body_hamil = ', input_file_two_body_hamil)
    print('num_electrons_up = ', num_electrons_up)
    print('num_electrons_down = ', num_electrons_down)
    print('num_orb = ', num_orb)
    print('num_k = ', num_k)
    print('D_TAU = ', D_TAU)
    print('NUM_WALKERS = ', NUM_WALKERS)
    print('NUM_STEPS = ', NUM_STEPS)
    print('Block_Divisor', block_divisor)
    print('UPDATE_METHOD = ', UPDATE_METHOD)
    print('REORTHO_PERIODICITY = ', REORTHO_PERIODICITY)
    print('REBAL_PERIODICITY = ', REBAL_PERIODICITY)
    print('output_file = ', output_file)
    return None

def read_datafile(filename):
    '''
    Read the data from the given file into a numpy array.
    '''
    return np.load(filename)


def generate_32bit_seed(base_seed, rank):
    """
    Generate a robust 32-bit seed using a hash function.

    Args:
        base_seed (int): The base seed for reproducibility.
        rank (int): The unique rank of the process (e.g., MPI rank).

    Returns:
        int: A 32-bit integer seed (0 <= seed < 2**32).
    """
    # Combine the base seed and rank into a unique string
    seed_str = f"{base_seed}_{rank}".encode()

    # Compute a hash (MD5 produces a 128-bit hash)
    hashed_value = hashlib.md5(seed_str).hexdigest()

    # Convert the hash into a 128-bit integer and reduce it to 32 bits
    seed_32bit = int(hashed_value, 16) % (2**32)

    return seed_32bit

#prime_multiplier = 982451653  # Large prime number
#seed = base_seed + rank * prime_multiplier
#
## Ensure seed fits within the valid range for PCG64DXSM
#seed = seed % (2**128)
#rng = np.random.Generator(np.random.PCG64DXSM(seed))
#local_seed = generate_32bit_seed(global_seed, rank)
#pcg64dxsm = np.random.PCG64DXSM(local_seed)
#rng = np.random.Generator(pcg64dxsm)
#np.random.seed(local_seed)

def reshape_H1(H1, num_k, num_orb):
    '''
    Reshape H1 (H1.npy shape is num_orb, num_orb, num_k   
    we implement the k_points to the H1)
    '''
    h1=np.zeros([num_orb*num_k,num_orb*num_k],dtype=np.complex128)
    for i in range(0,num_k):
        h1[i*num_orb:(i+1)*num_orb,i*num_orb:(i+1)*num_orb] = H1[:,:,i]
    for i in range(0,num_k):
        if first_cpu:
            print('H1 check?',np.allclose(h1[i*num_orb:(i+1)*num_orb,i*num_orb:(i+1)*num_orb] ,H1[:,:,i]))
    return h1

def reshape_H2(H2, num_k):
    '''
    generally the H2.npy (vasp output) has the shape of
    num_g * num_k ,num_orb * num_k, num_orb but we need
    num_orb * num_k, num_orb * num_k, ng'''
    h2 = H2.reshape(H2.shape[1], H2.shape[2] , H2.shape[0])
    return h2






def exp_Taylor(mat):
    OUT = np.eye(num_orb*num_k,dtype = 'complex128')
    C = np.eye(num_orb*num_k)
    for i in range (0,order_trunc):
      C = mat@C/(i+1)
      OUT += C
    return OUT

def overlap(left_slater_det, right_slater_det):
    '''
    It computes the overlap between two Slater determinants.
    '''
    overlap_mat = np.dot(left_slater_det.transpose(),right_slater_det)
    #overlap_mat = np.array([right_slater_det[num_orb*k:num_orb*k+num_electrons_up] for k in range(num_k)]).reshape(num_k*num_electrons_up,num_k*num_electrons_up)
    return overlap_mat

def get_q_list(q_list,q_selected):
    '''
    It returns q_list for a specicific momentum q_selected.
    '''
    return q_list[q_list[:,2]==q_selected]

def get_k1s(q_list,q_selected):
    '''
    It returns a list of k1s corresponding to q_selected.
    '''
    return get_q_list(q_list,q_selected)[:,0]

def get_k2s(q_list,q_selected):
    '''
    It returns a list of k2s corresponding to q_selected.
    '''

    return get_q_list(q_list,q_selected)[:,1]

def get_k1s_k2s(q_list,q_selected):
    '''
    It retuirns a list of tuples (k1s,k2s) corrsponding to q_selected.
    '''
    return list(zip(get_q_list(q_list,q_selected)[:,0],get_q_list(q_list,q_selected)[:,1]))

def get_A_k1_k2(h2,k1_idx,k2_idx):
    '''
    It returns the selected block of h2.  
    '''
    return h2[(k1_idx-1)*num_orb:k1_idx*num_orb,(k2_idx-1)*num_orb:k2_idx*num_orb,:]

def get_A_Q(h2,q_list,q_selected):
    Ql_selected = get_q_list(q_list,q_selected)
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    result = []
    for K1,K2 in K1s_K2s:
        result.append(get_A_k1_k2(h2,K1,K2))
    return np.array(result)

def comp_A_Q(h2,q_list,q_selected):
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    result = []
    for K1,K2 in K1s_K2s:
        result.append(get_A_k1_k2(h2,K1,K2))
    for K2,K1 in K1s_K2s:
        if K1!=K2:
            result.append(get_A_k1_k2(h2,K1,K2))
    return np.array(result)

def get_A_Q_full_mat(h2,q_list,q_selected):
    result = np.zeros_like(h2)
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    for K1,K2 in K1s_K2s:
        result[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:]=1
    return result*h2

def gen_A_e_Q(h2,q_list,q_selected):
    A_Q_full = get_A_Q_full_mat(h2,q_list,q_selected)
    a_e_Q = (A_Q_full + np.einsum('ijG->jiG', A_Q_full.conj()))/2
    return a_e_Q

def gen_A_o_Q(h2,q_list,q_selected):
    A_Q_full = get_A_Q_full_mat(h2,q_list,q_selected)
    a_o_Q = (A_Q_full - np.einsum('ijG->jiG', A_Q_full.conj()))*1j/2
    return a_o_Q

def gen_A_e_full(h2):
    '''
    It generates A_e.
    '''
    a_e = (h2 + np.einsum('ijG->jiG', h2.conj()))/2
    return a_e

def gen_A_o_full(h2):
    '''
    It generates A_o.
    '''
    a_o = (h2 - np.einsum('ijG->jiG', h2.conj()))*1j/2
    return a_o

def gen_Ae_Qs_list(h2,q_listl):
    ae_Qs = np.array([])
    for Q in range (1,num_k+1):
        ae_Qs = np.append(ae_Qs,gen_A_e_Q(h2,q_listl,Q))
    ae_Qs = ae_Qs.reshape([num_k,h2.shape[0],h2.shape[1],h2.shape[2]])
    return ae_Qs

def gen_Ao_Qs_list(h2,q_listl):
    ao_Qs = np.array([])
    for Q in range (1,num_k+1):
        ao_Qs = np.append(ao_Qs,gen_A_o_Q(h2,q_listl,Q))
    ao_Qs = ao_Qs.reshape([num_k,h2.shape[0],h2.shape[1],h2.shape[2]])
    return ao_Qs

def get_alpha_k1_k2(trial_0,h2,k1_idx,k2_idx):                                                               #####! it doesnt use mean field subtraction
    '''
    It returns alpha to be used as an intermediate object to compute one-body reduced density tensors.
    '''
    A_Q = get_A_k1_k2(h2,k1_idx,k2_idx)
    result = np.einsum('ip,prG->irG',trial_0.T,A_Q)
    return result

def get_alpha_Q(trial_0,h2,q_list,q_selected):
    Ql_selected = get_q_list(q_list,q_selected)
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    result = []
    for K1,K2 in K1s_K2s:
        result.append(get_alpha_k1_k2(trial_0,h2,K1,K2))
    return(np.array(result))

def get_alpha_Q_full_mat(trial,h2,q_list,q_selected):
    a_q_full_mat = get_A_Q_full_mat(h2,q_list,q_selected)
    return(np.einsum('ip,prG->irG',trial.T,a_q_full_mat))

def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial,walker)))

def avg_A_Q_new(trial,h2,q_list,q_selected):
    '''
    It returns average of two-body Hamiltonian for a specified Q.
    '''
    theta_full = theta(trial, trial)
    alpha_full = get_alpha_Q_full_mat(trial,h2,q_list,q_selected)
    f_full = np.einsum('nrG,rm->nmG', alpha_full,theta_full)
    tr_f = np.einsum('iiG->G',f_full)
    return 2*tr_f

def avg_A_Q(trial_0,trial,h2,q_list,q_selected):
    '''
    It returns average of two-body Hamiltonian for a specified Q.
    '''
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    theta_full = theta(trial, trial)
    h2_shape = h2.shape[2]
    result = 1j*np.zeros(h2_shape)
    for K1,K2 in K1s_K2s:
        alpha = get_alpha_k1_k2(trial_0,h2,K1,K2)
        result += np.einsum('iiG->G',np.einsum('nrG,rm->nmG',alpha,theta_full[(K2-1)*num_orb:K2*num_orb,(K1-1)*num_electrons_up:K1*num_electrons_up]))
    return 2*result

def A_af_MF_sub_new(trial,h2,q_list):
    avg_A_mat = np.zeros_like(h2)
    h2_shape = h2.shape
    for Q in range(1,2):
        avg_A_vec_Q = avg_A_Q_new(trial,h2,q_list,Q)
        K1s_K2s = get_k1s_k2s(q_list,Q)
        for K1,K2 in K1s_K2s:
            for g in range(h2_shape[2]):
                for r in range(num_orb):
                    avg_A_mat[(K1-1)*num_orb+r][(K2-1)*num_orb+r][g]=avg_A_vec_Q[g]
    return h2-avg_A_mat/num_electrons_up/2/num_k

def A_af_MF_sub(trial_0,trial,h2,q_list):
    '''
    It returns two body Hamiltonian after mean-field subtraction.
    '''
    avg_A_mat = np.zeros_like(h2)
    h2_shape = h2.shape
    for Q in range(1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0,trial,h2,q_list,Q)
        K1s_K2s = get_k1s_k2s(q_list,Q)
        for K1,K2 in K1s_K2s:
            for g in range(h2_shape[2]):
                for r in range(num_orb):
                    avg_A_mat[(K1-1)*num_orb+r][(K2-1)*num_orb+r][g]=avg_A_vec_Q[g]
    #print("H2subtraction", avg_A_mat)
    return h2-avg_A_mat/num_electrons_up/2/num_k

def H_1_mf(trial_0,trial,h2,h2_dagger,q_list,h1):
    '''
    It returns one-body part of the Hamiltonian after mean-field subtraction.
    '''
    change = 1j*np.zeros_like(h1)
    for Q in range(1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0,trial,h2,q_list,Q)                                                                      ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        avg_A_vec_Q_dagger = avg_A_Q(trial_0,trial,h2_dagger,q_list,Q)                                                        ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        K1s_K2s = get_k1s_k2s(q_list,Q)
        for K1,K2 in K1s_K2s:
            change[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb] = np.einsum('rpG->rp',np.einsum('G,rpG->rpG',avg_A_vec_Q_dagger,h2[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:])+np.einsum('G,rpG->rpG',avg_A_vec_Q,h2_dagger[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:]))
    #print("change H1",change)
    return h1+change/2

def H_0_mf(trial_0,trial,h2,h2_dagger,q_list):
    '''
    It resturns constant part of the Hamiltonian after mean-field subtraction.
    '''
    result = 0
    for Q in range (1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0, trial,h2,q_list,Q)
        print("avg_A_vec_Q", avg_A_vec_Q)
        avg_A_vec_Q_dagger = avg_A_Q(trial_0, trial,h2_dagger,q_list,Q)
    result +=  -(np.einsum('G->',avg_A_vec_Q*avg_A_vec_Q_dagger))/2/2/num_electrons_up/num_k
    #print("H-0", result)
    return result


def mean_field( h2, num_elec, num_band):
    mask = np.array(num_k * (num_elec * [True] + (num_band - num_elec) * [False]))
    m = np.sum(h2[mask,mask], axis=0)
    #m = np.einsum("iig->g", H2[mask][:,mask])
    #m = np.linalg.det( h2[:num_elec, : num_elec].T)
    return m


def gen_minus_q(q_list):
    qs=np.arange(1,num_k+1)
    minus_qs =[1]
    for q_selected in range (2,num_k+1):
        K1s_K2s = get_k1s_k2s(q_list,q_selected)
        for Q in range(2,num_k+1):
            K1s_K2s_comp = get_k1s_k2s(q_list,Q)
            flag = True
            for pair in K1s_K2s:
                if flag:
                    for pair_comp in K1s_K2s_comp:
                        if pair==swap(pair_comp):
                            flag=False
                            minus_qs.append(Q)
                            break
    return(np.array(minus_qs))

def get_alpha_full(trial,h2,q_list):
    out=[]
    for q_selected in range(1,num_k+1):
        out.append(get_alpha_Q_full_mat(trial,h2,q_list,q_selected))
    return(np.array(out))

def get_alpha_full_t(trial,h2_dagger,q_list,minus_q):
    out=[]
    for q_selected in range(1,num_k+1):
        out.append(get_alpha_Q_full_mat(trial,h2_dagger,q_list,minus_q[q_selected-1]))
    return(np.array(out))

def swap(a):
    return (a[1],a[0])



#@profile

def update_hyb(inputs,trial_0,trial,walker_mat,walker_weight,q_list,h_0,h_1,d_tau,e_0,h1_exp_half):
    NG = num_g
    new_walker_mat = np.zeros_like(walker_mat)
    new_walker_weight = np.zeros_like(walker_weight)
    theta_mat = []
    for i in range(NUM_WALKERS):
        theta_mat.append(theta(trial, walker_mat[i]))
    theta_mat=np.array(theta_mat)
    

    #fb_e_Q = -2j*SQRT_DTAU*contract('Nri,irG->NG',theta_mat, _e,optimize='greedy')
    #fb_o_Q = -2j*SQRT_DTAU*contract('Nri,irG->NG',theta_mat, alpha_full_o,optimize='greedy')
    if inputs.get('PRECISION') == 'Single':
        fb_e_Q = -2j*SQRT_DTAU*expr_fb_e(theta_mat).astype(np.complex64)
        fb_o_Q = -2j*SQRT_DTAU*expr_fb_o(theta_mat).astype(np.complex64)
    else: 
        fb_e_Q = -2j*SQRT_DTAU*expr_fb_e(theta_mat)
        fb_o_Q = -2j*SQRT_DTAU*expr_fb_o(theta_mat)
    ####Boundary condition for rare events  based on:https://doi.org/10.1103/PhysRevB.80.214116

    fb_e_Q[abs(fb_e_Q)>1] = 1.0
    fb_o_Q[abs(fb_o_Q)>1] = 1.0

   #####################################################################################################
    #x_e_Q = rng.random(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS)
    #x_o_Q = rng.random(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS)
    if inputs.get("PRECISION", "") == 'Single':
        x_e_Q = np.random.randn(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS).astype(np.float32)
        x_o_Q = np.random.randn(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS).astype(np.float32)
    else:
        x_e_Q = np.random.randn(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS)
        x_o_Q = np.random.randn(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS)


##########################################################################################################################
#    x_e_Q = np.random.randn(NG*NUM_WALKERS*comm.Get_size()).reshape(comm.Get_size(),NG,NUM_WALKERS)[comm.Get_rank()]
#    x_o_Q = np.random.randn(NG*NUM_WALKERS*comm.Get_size()).reshape(comm.Get_size(),NG,NUM_WALKERS)[comm.Get_rank()]
#    x_e_Q = np.random.randn(NG*NUM_WALKERS*comm.Get_size()).reshape(NG,NUM_WALKERS,comm.Get_size())[:,:,comm.Get_rank()]
#    x_o_Q = np.random.randn(NG*NUM_WALKERS*comm.Get_size()).reshape(NG,NUM_WALKERS,comm.Get_size())[:,:,comm.Get_rank()]
#    global first_call
#    if first_call:
#        first_call = False
#    else:
#        with open(f"size_{comm.Get_size()}_rank_{comm.Get_rank()}", "w") as f:
#            for i in range(NUM_WALKERS):
#                f.write(f"{comm.Get_size() * i + comm.Get_rank()} {x_e_Q[:,i]}\n")
#        comm.barrier()
#        exit()
##############################################################################################################################   
    h_2 = expr_h2_e(x_e_Q-fb_e_Q.T)+expr_h2_o(x_o_Q-fb_o_Q.T)  
    #h_2 = h_mf.two_body_e@(x_e_Q-fb_e_Q.T) + h_mf.two_body_o@(x_o_Q-fb_o_Q.T)
    print('H2 type', h_2.dtype) 
    for i in range(0,NUM_WALKERS):
        h= h_1 + SQRT_DTAU * 1j *h_2 [:,:,i]
        if inputs["PROPAG"] == 'Old':
            addend = walker_mat[i]
            for j in range(order_trunc+1):
                new_walker_mat[i] += addend
                addend = h@addend/(j + 1)

        
########       Different propagators Taylor, S1, and S2

        elif inputs["PROPAG"] == 'Taylor':
            prop_taylor = np.exp(d_tau * (-h_0 + e_0)) * exp_Taylor(h)
            new_walker_mat[i] = prop_taylor @ walker_mat[i]
        
        elif  inputs["PROPAG"]  == 'S1':
            
            prop_S1 = np.exp(d_tau * (-h_0 + e_0)) * expm(-d_tau * h_1) * exp_Taylor(SQRT_DTAU * 1j * h_2[:, :, i])
            new_walker_mat[i] = prop_S1 @ walker_mat[i]
        
        elif inputs["PROPAG"]  == 'S2':
            prop_S2 = h1_exp_half@exp_Taylor(SQRT_DTAU*1j*h_2[:,:,i])@h1_exp_half
            #prop_S2 = np.exp(d_tau * (-h_0 + e_0)) * expm(-d_tau * h_1 / 2) @ exp_Taylor(SQRT_DTAU * 1j * h_2[:, :, i]) @ expm(-d_tau * h_1 / 2)
            new_walker_mat[i] = prop_S2 @ walker_mat[i]
        
        else:
            raise ValueError("Invalid method selected. Choose from 'taylor', 'S1', or 'S2'.")

        ovrlap_ratio = np.linalg.det(overlap(trial,new_walker_mat[i]))**2 / np.linalg.det(overlap(trial,walker_mat[i]))**2
        ##ovrlap_ratio = np.linalg.det(overlap(trial,new_walker_mat[i])) / np.linalg.det(overlap(trial,walker_mat[i]))
        alpha = phase(ovrlap_ratio)

####new_weight

        new_walker_weight[i] = abs(ovrlap_ratio*(np.exp( np.dot(x_e_Q[:,i],fb_e_Q[i])-np.dot(fb_e_Q[i],fb_e_Q[i]/2))*np.exp(np.dot(x_o_Q[:,i], fb_o_Q[i])-np.dot(fb_o_Q[i],fb_o_Q[i]/2))))* max(0,np.cos(alpha))*walker_weight[i]
    return new_walker_mat, new_walker_weight


def propagator_fp(h_mf,h_1,d_tau,e_0):
    '''
    Free projection propagator.
    '''
  
    h_0 =h_mf.zero_body
#    print('H_0 from prop', h_0)
    Ae_list = h_mf.two_body_e
    Ao_list = h_mf.two_body_o
    NG = Ae_list.shape[2]
    x_e = np.random.randn(NG)
    x_o = np.random.randn(NG)
    h_2 =0#Ae_list@x_e + Ao_list@x_o
    return np.exp(-d_tau*(h_0-e_0))#*exp_Taylor(-d_tau*h_1+np.sqrt(d_tau)*1j*h_2)    



#@profile
#def measure_E_gs(trial,weights,walkers,h_1,h2,h2_dagger,q_list,minus_q,alpha_full,alpha_full_t,comm):
def measure_E_gs_old(trial,weights,walkers,h_1):#,comm):
    thetas=[]
    for i in range(NUM_WALKERS):
        thetas.append(theta(trial, walkers[i]))
    thetas=np.array(thetas)
    b=np.dot(trial.T,h_1)
    e1=0#2*contract('iNi->N',np.tensordot(b, thetas,axes=((1,1))))
    #fb_e_Q = 2*expr_fb_e(thetas)
    #fb_o_Q = 2*expr_fb_o(thetas) 
    #har_list_new=np.einsum('NG->N',fb_e_Q**2)+np.einsum('NG->N',fb_o_Q**2)
    #har_list = 2*expr_hart(thetas,thetas)
    har_list = 2*expr_hart_new(thetas,thetas)
    #print(har_list_new[0],har_list[0])
    #print('check hartree:', np.allclose(har_list,har_list_new,atol=1e-6))
    #exch_list = expr_exch_3(thetas,thetas)
    exch_list = expr_exch_new(thetas,thetas)
    #print('check exchange:', np.allclose(exch_list,exch_list_new,atol=1e-6))
    #exch_list=contract('Nri,qjrG,Npj,qipG->N',thetas,alpha_full,thetas,alpha_full_t)
    e_locs=e1+har_list-exch_list
    val=0
    for e_loc, weight in zip(e_locs, weights):
        val+=e_loc*weight
    comm.reduce(val)
    sum_w = np.sum(weights)
    comm.reduce(sum_w)
    return val/sum_w


def E_one(trial,weights,walkers,h_1):#,alpha_full,alpha_full_t):#,comm):
    thetas=[]
    for i in range(NUM_WALKERS):
        thetas.append(theta(trial, walkers[i]))
    thetas=np.array(thetas)
    b=np.dot(trial.T,h_1) 
    e1=2*contract('iNi->N',np.tensordot(b, thetas,axes=((1,1))))/num_k
    val=0
    for e_one, weight in zip(e1, weights):
        val+=e_one*weight/comm.Get_size()
    comm.reduce(val)
    sum_w = np.sum(weights/comm.Get_size())
    comm.reduce(sum_w)

    return val/sum_w




def Hartree(trial,weights,walkers):#,alpha_full,alpha_full_t):#,comm):
    thetas=[]
    for i in range(NUM_WALKERS):
        thetas.append(theta(trial, walkers[i]))
    thetas=np.array(thetas)
    har_list  =2*expr_hart_new_new(thetas,thetas) / num_k 
    val=0
    for hartree, weight in zip(har_list, weights):
        val+=hartree*weight/comm.Get_size()
    comm.reduce(val)
    sum_w = np.sum(weights/comm.Get_size())
    comm.reduce(sum_w)
    return val/sum_w

def Exchange(trial,weights,walkers):#,alpha_full,alpha_full_t):#,comm):
    thetas=[]
    for i in range(NUM_WALKERS):
        thetas.append(theta(trial, walkers[i]))
    thetas=np.array(thetas)
    exch_list =(-expr_exch_new_new(thetas,thetas) - num_electrons_up*num_k*fsg)/num_k
    val=0
    for exchange, weight in zip(exch_list, weights):
        val+=exchange*weight/comm.Get_size()
    comm.reduce(val)
    sum_w = np.sum(weights/comm.Get_size())
    comm.reduce(sum_w)
    return val/sum_w


def measure_E_gs_1(trial,weights,walkers,h_1,num_k):#,alpha_full,alpha_full_t):#,comm):
    e1=E_one(trial,weights,walkers,h_1)
    har_list  =Hartree(trial,weights,walkers) 
    exch_list =Exchange(trial,weights,walkers)
    e_locs=e1+har_list+exch_list
    return e_locs/num_k, e1/num_k, har_list/num_k, exch_list/num_k


def measure_E_gs(trial,weights,walkers,h_1,e_0):#,alpha_full,alpha_full_t):#,comm):
    thetas=[]
    for i in range(NUM_WALKERS):
        thetas.append(theta(trial, walkers[i]))
    thetas=np.array(thetas)
    b=np.dot(trial.T,h_1)
    e1=2*contract('iNi->N',np.tensordot(b, thetas,axes=((1,1)))) / num_k
    har_list  = 2* expr_hart_new_new(thetas,thetas) / num_k
    exch_list = (-expr_exch_new_new(thetas,thetas) - num_electrons_up * num_k *fsg)/num_k
    e_locs=e1+har_list+exch_list


    #if e_0!=0:
    #    e_locs[e_locs.real>e_0.real+np.sqrt(2./D_TAU)]=e_0+np.sqrt(2./D_TAU)
    #    e_locs[e_locs.real<e_0.real-np.sqrt(2./D_TAU)]=e_0-np.sqrt(2./D_TAU)

#    with open(f"energy_size_{comm.Get_size()}_rank_{comm.Get_rank()}", "w") as f:
#        for i, (e_loc, weight) in enumerate(zip(e_locs, weights)):
#            f.write(f"{i} {e_loc} {weight / comm.Get_size()}\n")
#    global first_call
#    if not first_call:
#        comm.barrier()
#        exit()
#    first_call = False
    val=0
    for e_loc, weight in zip(e_locs, weights):
        val+=e_loc*weight
    #print('val before', val, comm.Get_rank())
    val_glb= comm.allreduce(val)/comm.Get_size()
    #if comm.Get_rank()==0:
    #print('val after', val_glb, comm.Get_rank())
    sum_w = np.sum(weights)
    #print('sum before', sum_w, comm.Get_rank())
    sum_glb= comm.allreduce(sum_w)/comm.Get_size()
    #print('sum after', sum_glb, comm.Get_rank())
    return val_glb/sum_glb, val_glb, sum_glb

def update_fp(trial_0,trial,walker_mat,walker_weight,q_list,h_mf,h_1,d_tau,e_0,h_original,h2_dagger,alpha_full_e,alpha_full_o):
    h_0 = h_mf.zero_body
    h_2 = 1j*np.zeros([num_orb*num_k,num_orb*num_k,NUM_WALKERS])
    NG = h_mf.two_body_e.shape[2]
    new_walker_mat = np.zeros_like(walker_mat)
    new_walker_weight = np.zeros_like(walker_weight)
    theta_mat = []
    for i in range(NUM_WALKERS):
        theta_mat.append(theta(trial, walker_mat[i]))
    theta_mat=np.array(theta_mat)
    x_e_Q = np.random.randn(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS)
    x_o_Q = np.random.randn(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS)
    h_2 = h_2 + h_mf.two_body_e@(x_e_Q) + h_mf.two_body_o@(x_o_Q)
    for i in range(0,NUM_WALKERS):
        propagator = np.exp(D_TAU*(e_0-h_0))*exp_Taylor(h_1+SQRT_DTAU*1j*h_2[:,:,i])
        new_walker_mat[i] = np.dot(propagator,walker_mat[i])
        ovrlap_ratio = np.linalg.det(overlap(trial,new_walker_mat[i]))**2 / np.linalg.det(overlap(trial,walker_mat[i]))**2
        new_walker_weight[i] = ovrlap_ratio * walker_weight[i]
    return new_walker_mat, new_walker_weight

def propagator_fp_noMF(a_e,a_o,h_1,d_tau,e_0):
    h_2 = np.zeros([num_orb*num_k,num_orb*num_k])
    NG = a_e[0].shape[2]
    for q_selected in range (1,num_k+1):
        x_e = np.random.randn(NG)
        x_o = np.random.randn(NG)
        h_2 = h_2 + a_e[q_selected-1]@x_e + a_o[q_selected-1]@x_o
    return np.exp(d_tau*e_0)*exp_Taylor(h_1+np.sqrt(d_tau)*1j*h_2)

def expect(propag, psi, walker):
    '''
    It computes the average of propag (i.e., exp(A)) over trial wave function psi.
    '''
    result = np.dot(propag,walker)
    result = np.dot(psi.conj().transpose(),result)
    return (np.linalg.det(result))**2

def average_Hamil(h_mf, h_1, psi_up, walker, n_w, d_tau, e_0, n):
    '''
    It computes 1/n_w Re { sum_{w} ( 1 - det(psi_up_dagger . propog . psi_up ) x det(psi_down_dagger . propog . psi_down ) ) / d_tau }.
    '''
    avg = 0
    for _ in range (0,n_w):
        propog = propagator_fp(h_mf, h_1, d_tau, e_0)
        det_up = expect(propog,psi_up,walker)
        avg += ((1-det_up)/d_tau/n).real
    avg = avg/n_w
    return avg


def reortho_qr(walker_matrix):
    '''
    It uses QR decomposition to reorthogonalize the orbitals in a single walker.
    '''
    Q, R = np.linalg.qr(walker_matrix)
    return Q

def rebalance_comb(weights):
    '''
    It rebalances weights using Eq. (A2) of Phys. Rev. E 80, 046704 (2009); we set M=N.
    '''
    wt = np.sum(weights)
    c = np.cumsum(weights)
    d=np.append([0],c)
    res = wt/len(weights)
    eps = -np.random.random()
    new_walkers_indices=np.array([])
    for j in range(1,len(d)):
        for k in range(1,len(d)):
            if (k+eps)*res>d[j]:
                 break
            if ( (k+eps)*res<=d[j] and (k+eps)*res>d[j-1]):
                new_walkers_indices=np.append(new_walkers_indices,j-1)
    return new_walkers_indices.astype(int)

def init_walker_mats(n_walkers,init_mat):
    return np.array(n_walkers * [init_mat], dtype=np.complex128)

def init_walkers_weights(n_walkers):
    '''
    It initializes the weights.
    '''
    return np.ones(n_walkers, dtype=np.complex128)

def blockAverage(datastream, block_divisor):
    Nobs = len(datastream)
    N = block_divisor
    minBlockSize = 1;
    maxBlockSize = int(Nobs/N);
    NumBlocks = maxBlockSize - minBlockSize
    blockMean = np.zeros(NumBlocks)
    blockVar = np.zeros(NumBlocks)
    blockCtr = 0
    for blockSize in range(minBlockSize, maxBlockSize):
        Nblock = int(Nobs/blockSize)
        obsProp = np.zeros(Nblock)
        for i in range(1,Nblock+1):
            ibeg = (i-1) * blockSize
            iend =  ibeg + blockSize
            obsProp[i-1] = np.mean(datastream[ibeg:iend])
        blockMean[blockCtr] = np.mean(obsProp)
        blockVar[blockCtr] = (np.var(obsProp)/(Nblock - 1))
        blockCtr += 1
    v = np.arange(minBlockSize,maxBlockSize)
    return v, blockVar, blockMean

def set_update_method(method):
    '''
    The *method* is a user provided string read from the input file.
    '''
    if method[0]=="H" or method[0]=="h":
        return update_hyb
    elif method[0]=="F" or method[0]=="f" :
        return update_fp
    else:
        return update_hyb

def update_walker(trial_0,trial,walker_mat,walker_weight,q_list,h_0,h_1,d_tau,e_0,selected_update_method):
    return selected_update_method(trial_0,trial,walker_mat,walker_weight,q_list,h_0,h_1,d_tau,e_0,propagator)

def h2_q_name(q_selected):
    if q==1:
        return h2_1
    if q==2:
        return h2_2
    if q==3:
        return h2_3
    if q==4:
        return h2_4
    if q==5:
        return h2_5
    if q==6:
        return h2_6
    if q==7:
        return h2_7
    if q==8:
        return h2_8
