from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() != 0:
    print = lambda *arg, **kwargs: None

import numpy as np
from dataclasses import dataclass
from afqmc_ref_INPUT import *
from opt_einsum import contract_expression
from opt_einsum import contract
from time import time
from scipy.linalg import expm

#from afqmc_ref_funcs import ( 
#    A_af_MF_sub, 
#    get_k1s_k2s,
#    get_q_list,
#    gen_minus_q,
#    swap,
#    overlap,
#    theta,
#    get_alpha_Q_full_mat,
#    gen_A_full,  # Using gen_A_full instead of gen_A_e_full and gen_A_o_full
#    read_datafile, 
#    Hamiltonian
#)

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
def read_datafile(filename):
    '''
    Read the data from the given file into a numpy array.
    '''
    return np.load(filename)

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





def gen_minus_q(q_list):
    """
    Optimized function to generate minus_q values more efficiently.
    """
    qs = np.arange(1, num_k + 1)
    minus_qs = [1]

    for q_selected in range(2, num_k + 1):
        K1s_K2s = get_k1s_k2s(q_list, q_selected)
        seen_swaps = {swap(pair) for pair in K1s_K2s}  # Store swapped pairs in a set

        for Q in range(2, num_k + 1):
            K1s_K2s_comp = get_k1s_k2s(q_list, Q)

            # Check if any swapped pair exists in K1s_K2s_comp
            if any(pair in seen_swaps for pair in K1s_K2s_comp):
                minus_qs.append(Q)
                break  # Break early since Q is already matched

    return np.array(minus_qs)

def swap(a):
    """ Optimized swap function using tuple slicing. """
    return a[::-1]


def get_alpha_k1_k2(trial_0,h2,k1_idx,k2_idx):                                                               #####! it doesnt use mean field subtraction
    '''
    It returns alpha to be used as an intermediate object to compute one-body reduced density tensors.
    '''
    A_Q = get_A_k1_k2(h2,k1_idx,k2_idx)
    result = np.einsum('ip,prG->irG',trial_0.T,A_Q)
    return result

def get_A_k1_k2(h2,k1_idx,k2_idx):
    '''
    It returns the selected block of h2.  
    '''
    return h2[(k1_idx-1)*num_orb:k1_idx*num_orb,(k2_idx-1)*num_orb:k2_idx*num_orb,:]

def get_alpha_full(trial, h2, q_list):
    """
    Optimized function to compute full alpha matrices without list-to-array conversion overhead.
    """
    out = np.empty((num_k, *h2.shape), dtype=h2.dtype)  # Pre-allocate array

    for q_selected in range(num_k):  # NumPy indices are 0-based
        out[q_selected] = get_alpha_Q_full_mat(trial, h2, q_list, q_selected + 1)

    return out

def get_alpha_full_t(trial, h2_dagger, q_list, minus_q):
    """
    Optimized version of get_alpha_full_t with reduced indexing overhead.
    """
    out = np.empty((num_k, *h2_dagger.shape), dtype=h2_dagger.dtype)  # Pre-allocate array

    for q_selected, q_value in enumerate(minus_q):
        out[q_selected] = get_alpha_Q_full_mat(trial, h2_dagger, q_list, q_value)

    return out


def get_alpha_Q_full_mat(trial,h2,q_list,q_selected):
    a_q_full_mat = get_A_Q_full_mat(h2,q_list,q_selected)
    return(np.einsum('ip,prG->irG',trial.T,a_q_full_mat))

def get_A_Q_full_mat(h2, q_list, q_selected):
    """ Efficient in-place update of the selected Hamiltonian matrix. """
    for K1, K2 in get_k1s_k2s(q_list, q_selected):
        h2[(K1-1)*num_orb:K1*num_orb, (K2-1)*num_orb:K2*num_orb, :] *= 1  # In-place modification
    return h2

def get_k1s_k2s(q_list,q_selected):
    '''
    It retuirns a list of tuples (k1s,k2s) corrsponding to q_selected.
    '''
    return list(zip(get_q_list(q_list,q_selected)[:,0],get_q_list(q_list,q_selected)[:,1]))

def get_q_list(q_list,q_selected):
    '''
    It returns q_list for a specicific momentum q_selected.
    '''
    return q_list[q_list[:,2]==q_selected]


def mean_field(h2, num_elec, num_band):
    """
    Computes mean-field energy from the two-body Hamiltonian.
    """
    # Optimized mask calculation
    mask = np.repeat([True] * num_elec + [False] * (num_band - num_elec), num_k)

    # Efficient summation without redundant slicing
    return np.sum(h2[mask, mask], axis=0)
def H_0_mf(trial_0, trial, h2, h2_dagger, q_list):
    """
    Returns constant part of the Hamiltonian after mean-field subtraction.
    """
    # Compute avg_A only once for the final Q value (loop is redundant)
    avg_A_vec_Q = avg_A_Q(trial_0, trial, h2, q_list, num_k)
    avg_A_vec_Q_dagger = avg_A_Q(trial_0, trial, h2_dagger, q_list, num_k)

    # More efficient sum operation instead of einsum
    result = - (np.sum(avg_A_vec_Q * avg_A_vec_Q_dagger)) / (2 * 2 * num_electrons_up * num_k)

def H_1_mf(trial_0, trial, h2, h2_dagger, q_list, h1):
    """
    Optimized one-body part of the Hamiltonian after mean-field subtraction.
    """
    change = np.zeros_like(h1, dtype=np.complex128)  # More efficient memory allocation

    # Precompute all avg_A_Q values to avoid redundant calls
    avg_A_vecs_Q = [avg_A_Q(trial_0, trial, h2, q_list, Q) for Q in range(1, num_k + 1)]
    avg_A_vecs_Q_dagger = [avg_A_Q(trial_0, trial, h2_dagger, q_list, Q) for Q in range(1, num_k + 1)]

    for Q, avg_A_vec_Q, avg_A_vec_Q_dagger in zip(range(1, num_k + 1), avg_A_vecs_Q, avg_A_vecs_Q_dagger):
        K1s_K2s = get_k1s_k2s(q_list, Q)

        for K1, K2 in K1s_K2s:
            h2_block = h2[(K1-1)*num_orb:K1*num_orb, (K2-1)*num_orb:K2*num_orb, :]
            h2_dagger_block = h2_dagger[(K1-1)*num_orb:K1*num_orb, (K2-1)*num_orb:K2*num_orb, :]

            # Optimized einsum contraction
            change[(K1-1)*num_orb:K1*num_orb, (K2-1)*num_orb:K2*num_orb] = np.einsum(
                'G,rpG->rp', avg_A_vec_Q_dagger, h2_block
            ) + np.einsum(
                'G,rpG->rp', avg_A_vec_Q, h2_dagger_block
            )

    return h1 + change / 2

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


def A_af_MF_sub_new(trial, h2, q_list):
    """
    Optimized version of A_af_MF_sub_new with NumPy vectorization.
    """
    h2_shape = h2.shape
    avg_A_mat = np.zeros_like(h2)

    # Q is always 1, so we remove the loop
    Q = 1
    avg_A_vec_Q = avg_A_Q_new(trial, h2, q_list, Q)
    K1s_K2s = get_k1s_k2s(q_list, Q)

    # Vectorized assignment
    K1_indices = np.array([(K1-1) * num_orb for K1, _ in K1s_K2s])
    K2_indices = np.array([(K2-1) * num_orb for _, K2 in K1s_K2s])

    avg_A_mat[K1_indices[:, None] + np.arange(num_orb),
              K2_indices[:, None] + np.arange(num_orb),
              :] = avg_A_vec_Q[np.newaxis, np.newaxis, :]

    return h2 - avg_A_mat / (num_electrons_up * 2 * num_k)

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
def gen_A_full(h2):
    """
    Generates both A_e and A_o efficiently.
    """
    h2_T = h2.conj().transpose(1, 0, 2)  # More efficient than einsum

    a_e = (h2 + h2_T) / 2
    a_o = (h2 - h2_T) * 1j / 2
    print('a-e', a_e.shape)
    return a_e, a_o

def overlap(left_slater_det, right_slater_det):
    '''
    It computes the overlap between two Slater determinants.
    '''
    ##overlap_mat = np.dot(left_slater_det.transpose(),right_slater_det)
    #overlap_mat = np.array([right_slater_det[num_orb*k:num_orb*k+num_electrons_up] for k in range(num_k)]).reshape(num_k*num_electrons_up,num_k*num_electrons_up)
    return left_slater_det.T @ right_slater_det


def theta(trial, walker):
    """
    Optimized computation of theta using np.linalg.solve to avoid explicit inversion.
    """
    return np.linalg.solve(overlap(trial, walker).T, walker.T).T

# ✅ Load Hamiltonian directly without extra copies
hamil = HAMILTONIAN
hamil_MF = HAMILTONIAN_MF
ql = np.array(read_datafile(q_list), order='C').T
hamil.two_body = np.array(read_datafile(input_file_two_body_hamil), dtype=np.complex128)
h2_t = np.einsum('prG->rpG', hamil.two_body.conj())
m_q = gen_minus_q(ql)
H1 = np.array(read_datafile(input_file_one_body_hamil),dtype=np.complex128)
h1 = reshape_H1(H1, num_k, num_orb)
hamil.one_body = h1

# ✅ Mean-field subtraction
h2_af_MF_sub = A_af_MF_sub(PSI_T_up_0, PSI_T_up, hamil.two_body, ql)
h_self = -contract('ijG,jkG->ik',hamil.two_body,h2_t)/2#/num_k
hamil_MF.zero_body =  H_0_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql)
hamil_MF.one_body = H_1_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql,hamil.one_body)


H_1_self = -D_TAU * (hamil_MF.one_body+h_self)
#H1_self_exp = expm(H_1_self)
H1_self_half_exp = expm(H_1_self/2)

print('Hamiltonian done')
# ✅ Compute gen_A_full once (returns a_e and a_o)
A_e, A_o = gen_A_full(h2_af_MF_sub)

# ✅ Compute alpha values efficiently
ALPHA_E = contract('ip,prG->irG', PSI_T_up.T, A_e)  # Use a_e for ALPHA_E
ALPHA_O = contract('ip,prG->irG', PSI_T_up.T, A_o)  # Use a_o for ALPHA_O

# ✅ Compute alp and alp_t using opt_einsum
alp = contract('pi,prG->irG', PSI_T_up, hamil.two_body)
alp_t = contract('pi,rpG->irG', PSI_T_up, hamil.two_body.conj())

# ✅ Define optimized contractions using correct A_e and A_o
expr_h2_e = contract_expression('ijG,GN->ijN', A_e, (num_g, NUM_WALKERS), constants=[0], optimize='greedy')
expr_h2_o = contract_expression('ijG,GN->ijN', A_o, (num_g, NUM_WALKERS), constants=[0], optimize='greedy')
expr_fb_e = contract_expression('Nri,irG->NG', (NUM_WALKERS, num_orb*num_k, num_electrons_up*num_k), ALPHA_E, constants=[1], optimize='greedy')
expr_fb_o = contract_expression('Nri,irG->NG', (NUM_WALKERS, num_orb*num_k, num_electrons_up*num_k), ALPHA_O, constants=[1], optimize='greedy')
expr_exch_new_new = contract_expression('Nri,jrG,Npj,ipG->N', (NUM_WALKERS, num_orb*num_k, num_electrons_up*num_k), alp, (NUM_WALKERS, num_orb*num_k, num_electrons_up*num_k), alp_t, constants=[1,3], optimize='greedy')
expr_hart_new_new = contract_expression('Nri,irG,Npj,jpG->N', (NUM_WALKERS, num_orb*num_k, num_electrons_up*num_k), alp, (NUM_WALKERS, num_orb*num_k, num_electrons_up*num_k), alp_t, constants=[1,3], optimize='greedy')


print('Alpha file is done!')
#H2 = np.array(read_datafile(input_file_two_body_hamil),dtype=np.complex128)
#h2 = H2#np.moveaxis(H2, 0, -1) 
#hamil.two_body = h2
#h2_t = np.einsum('prG->rpG', hamil.two_body.conj())
#
#
#ql = np.array(read_datafile(q_list),order='C').T
#m_q = gen_minus_q(ql)
#print('Hamiltonian done')
#
##alpha_mul_new = get_alpha_mul(PSI_T_up,hamil.two_body,h2_t,ql,m_q)
#
##alp = get_alpha_full(PSI_T_up,hamil.two_body,ql)
##np.save('alpha_intl.npy',alp)
##print('alpha done')
##alp_t = get_alpha_full_t(PSI_T_up,h2_t,ql,m_q)
#print('PSI_T_up', PSI_T_up.shape)
#print('hamil_two ',hamil.two_body.shape)
#alp = np.einsum('pi, prg -> irg',PSI_T_up , hamil.two_body)
#alp_t = np.einsum('pi, rpg -> irg', PSI_T_up, hamil.two_body.conj())
##print('alpha_t done')
##np.save('alpha_t.npy',alp_t)
##alp = np.load('alpha.npy')
##alp_t = np.load('alpha_t.npy')
##print('alps -> ', alp.itemsize, alp_t.itemsize)
#print('alps done')
#
##alpha_mul_old = contract('qipG,qjrG->ipjr',ALPHA_FULL,ALPHA_FULL_T)
#
##print('???????', np.allclose(alpha_mul_old,alpha_mul_new))
##print()
##print()
##print()
#
#hamil_MF = HAMILTONIAN_MF
##st_time=time()
#h2_af_MF_sub = A_af_MF_sub(PSI_T_up_0,PSI_T_up,hamil.two_body,ql)
###h2_af_MF_sub = np.load('H2_af.npy') #A_af_MF_sub(PSI_T_up_0,PSI_T_up,hamil.two_body,ql)
###np.save('H2_af_intl.npy',h2_af_MF_sub)
##print('HF time = ', time()-st_time)
##print('HF done')
##st_time=time()
#
##hamil_MF.zero_body = H_0_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql)
#hamil_MF.zero_body = mean_field(h2, num_electrons_up, num_orb)
##
##print('h0 = ', hamil_MF.zero_body)
#hamil_MF.one_body = H_1_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql,hamil.one_body)
#hamil_MF.two_body_e = gen_A_e_full(h2_af_MF_sub)
#
#hamil_MF.two_body_o = gen_A_o_full(h2_af_MF_sub)
##print('h_e -> ', hamil_MF.two_body_e.itemsize, hamil_MF.two_body_o.itemsize)
##del(hamil)
#
#ALPHA_E = contract('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_e)
##np.save('a_e_intl.npy',ALPHA_E)
#ALPHA_O = contract('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_o)
##np.save('a_o_intl.npy',ALPHA_O)
##ALPHA_E = np.load('a_e.npy') #np.einsum('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_e)
##ALPHA_O = np.load('a_o.npy') #np.einsum('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_o)
##print('even odd time intl = ', time()-st_time)
##print('even odd done')
##st_time=time()
##ALPHAs_MUL = np.load('aa.npy')#get_alpha_mul(PSI_T_up,hamil.two_body,h2_t,ql,m_q)
##print()
##print('ALPHAs_MUL time = ', time()-st_time)
##print('aat done')
##np.save('aa.npy',ALPHAs_MUL)
#
#
##expr_exch_3 = contract_expression('Nri,qjrG,Npj,qipG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL_T,constants=[1,3],optimize='greedy')
##st_time=time()
##expr_exch_new = contract_expression('Nri,Npj,jrip->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHAs_MUL,constants=[2],optimize='greedy')
##print('expr_exch_new time = ', time()-st_time)
##expr_hart = contract_expression('Nri,qirG,Npj,qjpG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL_T,constants=[1,3],optimize='greedy')
##st_time=time()
##expr_hart_new = contract_expression('Nri,Npj,irjp->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHAs_MUL,constants=[2],optimize='greedy')
##expr_hart_new_new = contract_expression('Nri,irG,Npj,jpG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_t,constants=[1,3],optimize='greedy')
##print('expr_hart_new time = ', time()-st_time)
#expr_h2_e = contract_expression('ijG,GN->ijN',hamil_MF.two_body_e,(num_g,NUM_WALKERS),constants=[0],optimize='greedy')
#expr_h2_o = contract_expression('ijG,GN->ijN',hamil_MF.two_body_o,(num_g,NUM_WALKERS),constants=[0],optimize='greedy')
#expr_fb_e = contract_expression('Nri,irG->NG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_E,constants=[1],optimize='greedy')
#expr_fb_o = contract_expression('Nri,irG->NG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_O,constants=[1],optimize='greedy')
#
#expr_exch_new_new = contract_expression('Nri,jrG,Npj,ipG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_t,constants=[1,3],optimize='greedy')
#expr_hart_new_new = contract_expression('Nri,irG,Npj,jpG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_t,constants=[1,3],optimize='greedy')
#      
##exp_f = contract_expression('Nri,qirG->NqG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL,constants=[1])
##exp_ft = contract_expression('Nri,qirG->NqG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL_T,constants=[1])
##expr_exch = contract_expression('Nri,jrG,Npj,ipG->N', (NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k), (num_electrons_up*num_k,num_orb*num_k,),(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(num_electrons_up*num_k,num_orb*num_k,num_g))#,optimize='greedy')
##expr_exch_2 = contract_expression('Nri,qjrG,Npj,qipG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(num_k,num_electrons_up*num_k,num_orb*num_k,num_g),(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(num_k,num_electrons_up*num_k,num_orb*num_k,num_g))
