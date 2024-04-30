import numpy as np
from dataclasses import dataclass
from afqmc_INPUT_H2_0 import *
from opt_einsum import contract_expression
from opt_einsum import contract
from time import time

@dataclass
class HAMILTONIAN:
    one_body: np.complex64
    two_body: np.complex64


def read_datafile(filename):
    '''
    Read the data from the given file into a numpy array.
    '''
    return np.load(filename)

def swap(a):
    return (a[1],a[0])

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

def get_alpha_k1_k2(trial_0,h2,k1_idx,k2_idx):
    '''
    It returns alpha to be used as an intermediate object to compute one-body reduced density tensors.
    '''
    A_Q = get_A_k1_k2(h2,k1_idx,k2_idx)
    result = contract('ip,prG->irG',trial_0.T,A_Q)
    return result

def get_A_k1_k2(h2,k1_idx,k2_idx):
    '''
    It returns the selected block of h2.
    '''
    return h2[(k1_idx-1)*num_orb:k1_idx*num_orb,(k2_idx-1)*num_orb:k2_idx*num_orb,:]

def get_alpha_full(trial,h2,q_list):
    out=np.zeros([num_k*num_electrons_up,num_orb*num_k,num_g],dtype=np.complex64)
    for q_selected in range(1,num_k+1):
        out+=get_alpha_Q_full_mat(trial,h2,q_list,q_selected)
    return out

def get_alpha_full_t(trial,h2_dagger,q_list,minus_q):
    out=np.zeros([num_k*num_electrons_up,num_orb*num_k,num_g],dtype=np.complex64)
    for q_selected in range(1,num_k+1):
        out+=get_alpha_Q_full_mat(trial,h2_dagger,q_list,minus_q[q_selected-1])
    return out

def get_alpha_Q_full_mat(trial,h2,q_list,q_selected):
    a_q_full_mat = get_A_Q_full_mat(h2,q_list,q_selected)
    return(contract('ip,prG->irG',trial.T,a_q_full_mat))

#@profile
def get_A_Q_full_mat(h2,q_list,q_selected):
    result = np.zeros_like(h2)
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    for K1,K2 in K1s_K2s:
        #result[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:]=1
        result[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:]=h2[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:]
    #return result*h2
    return result   
'''
#@profile
def get_alpha_mul(trial,h2,h2_dagger,q_list,minus_q):
    aat = np.zeros([num_k*num_electrons_up,num_orb*num_k,num_k*num_electrons_up,num_orb*num_k],dtype=np.complex64)
    print(aat.shape)
    print('allocated memory for aat = ', aat.size*aat.itemsize/2**30 )
    for q_selected in range(1,num_k+1):
        print('q_selected = ', q_selected)
        st_time=time()
        alpha_q   = get_alpha_Q_full_mat(trial,h2,q_list,q_selected)
        alpha_q_t = get_alpha_Q_full_mat(trial,h2_dagger,q_list,minus_q[q_selected-1])
        print('slicing time = ', time()-st_time)
        #print('alpha_q.shape = ', alpha_q.shape)
        #print('alpha_q_t.shape = ' , alpha_q_t.shape)
        st_time=time()
        aat+=contract('ipG,jrG->ipjr',alpha_q,alpha_q_t)
        print('contract time = ', time()-st_time)
    return(aat)
'''

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

@dataclass
class HAMILTONIAN_MF:
    zero_body: np.complex64
    one_body: np.complex64
    two_body_e: np.complex64
    two_body_o: np.complex64

def H_0_mf(trial_0,trial,h2,h2_dagger,q_list):
    '''
    It resturns constant part of the Hamiltonian after mean-field subtraction.
    '''
    result = 0
    for Q in range (1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0,trial,h2,q_list,Q)
        avg_A_vec_Q_dagger = avg_A_Q(trial_0,trial,h2_dagger,q_list,Q)
        result +=  -(contract('G->',avg_A_vec_Q*avg_A_vec_Q_dagger))/2/2/num_electrons_up/num_k
    return result

def H_1_mf(trial_0,trial,h2,h2_dagger,q_list,h1):
    '''
    It returns one-body part of the Hamiltonian after mean-field subtraction.
    '''
    change = 1j*np.zeros_like(h1)
    for Q in range(1,num_k+1):
        avg_A_vec_Q = avg_A_Q(trial_0,trial,h2,q_list,Q)
        avg_A_vec_Q_dagger = avg_A_Q(trial_0,trial,h2_dagger,q_list,Q)
        K1s_K2s = get_k1s_k2s(q_list,Q)
        for K1,K2 in K1s_K2s:
            change[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb] = contract('rpG->rp',contract('G,rpG->rpG',avg_A_vec_Q_dagger,h2[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:])+contract('G,rpG->rpG',avg_A_vec_Q,h2_dagger[(K1-1)*num_orb:K1*num_orb,(K2-1)*num_orb:K2*num_orb,:]))
    return h1+change/2

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
    return h2-avg_A_mat/num_electrons_up/2/num_k

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
        result += contract('iiG->G',contract('nrG,rm->nmG',alpha,theta_full[(K2-1)*num_orb:K2*num_orb,(K1-1)*num_electrons_up:K1*num_electrons_up]))
    return 2*result

def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial,walker)))

def overlap(left_slater_det, right_slater_det):
    '''
    It computes the overlap between two Slater determinants.
    '''
    #overlap_mat = np.dot(left_slater_det.transpose(),right_slater_det)
    overlap_mat = np.array([right_slater_det[num_orb*k:num_orb*k+num_electrons_up] for k in range(num_k)]).reshape(num_k*num_electrons_up,num_k*num_electrons_up)
    return overlap_mat


hamil = HAMILTONIAN
h1 = np.array(read_datafile(input_file_one_body_hamil),order='C')
H1=np.zeros([num_orb*num_k,num_orb*num_k],dtype=np.complex64)
for i in range(0,num_k):
    H1[i*num_orb:(i+1)*num_orb,i*num_orb:(i+1)*num_orb] = h1[:,:,i]
hamil.one_body = H1
print('h1 -> ', hamil.one_body.itemsize)
del(h1)
del(H1)
hamil.two_body = np.array(read_datafile(input_file_two_body_hamil),dtype=np.complex64)
print('h2 -> ', hamil.two_body.itemsize)
h2_t = np.einsum('rpG->prG', hamil.two_body.conj())
print('h2_t -> ', h2_t.itemsize)
ql = np.array(read_datafile(q_list),order='C').T
m_q = gen_minus_q(ql)
print('Hamiltonian done')

#alpha_mul_new = get_alpha_mul(PSI_T_up,hamil.two_body,h2_t,ql,m_q)

alp = get_alpha_full(PSI_T_up,hamil.two_body,ql)
np.save('alpha.npy',alp)
print('alpha done')
alp_t = get_alpha_full_t(PSI_T_up,h2_t,ql,m_q)
print('alpha_t done')
np.save('alpha_t.npy',alp_t)
#alp = np.load('alpha_64.npy')
#alp_t = np.load('alpha_t_64.npy')
print('alps -> ', alp.itemsize, alp_t.itemsize)
print('alps done')

#alpha_mul_old = contract('qipG,qjrG->ipjr',ALPHA_FULL,ALPHA_FULL_T)

#print('???????', np.allclose(alpha_mul_old,alpha_mul_new))
#print()
#print()
#print()

hamil_MF = HAMILTONIAN_MF
st_time=time()
h2_af_MF_sub = A_af_MF_sub(PSI_T_up_0,PSI_T_up,hamil.two_body,ql)
#h2_af_MF_sub = np.load('H2_af_64.npy') #A_af_MF_sub(PSI_T_up_0,PSI_T_up,hamil.two_body,ql)
np.save('H2_af.npy',h2_af_MF_sub)
print('HF time = ', time()-st_time)
print('HF done')
st_time=time()
hamil_MF.zero_body = H_0_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql)
print('h0 = ', hamil_MF.zero_body)
hamil_MF.one_body = H_1_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql,hamil.one_body)
hamil_MF.two_body_e = gen_A_e_full(h2_af_MF_sub)
hamil_MF.two_body_o = gen_A_o_full(h2_af_MF_sub)
print('h_e -> ', hamil_MF.two_body_e.itemsize, hamil_MF.two_body_o.itemsize)
del(hamil)

ALPHA_E = contract('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_e)
np.save('a_e.npy',ALPHA_E)
ALPHA_O = contract('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_o)
np.save('a_o.npy',ALPHA_O)
#ALPHA_E = np.load('a_e_64.npy') #np.einsum('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_e)
#ALPHA_O = np.load('a_o_64.npy') #np.einsum('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_o)
print('even odd time = ', time()-st_time)
print('even odd done')
#st_time=time()
#ALPHAs_MUL = np.load('aa.npy')#get_alpha_mul(PSI_T_up,hamil.two_body,h2_t,ql,m_q)
#print()
#print('ALPHAs_MUL time = ', time()-st_time)
#print('aat done')
#np.save('aa.npy',ALPHAs_MUL)


#expr_exch_3 = contract_expression('Nri,qjrG,Npj,qipG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL_T,constants=[1,3],optimize='greedy')
#st_time=time()
#expr_exch_new = contract_expression('Nri,Npj,jrip->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHAs_MUL,constants=[2],optimize='greedy')
#print('expr_exch_new time = ', time()-st_time)
#expr_hart = contract_expression('Nri,qirG,Npj,qjpG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL_T,constants=[1,3],optimize='greedy')
#st_time=time()
#expr_hart_new = contract_expression('Nri,Npj,irjp->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHAs_MUL,constants=[2],optimize='greedy')
#print('expr_hart_new time = ', time()-st_time)
expr_h2_e = contract_expression('ijG,GN->ijN',hamil_MF.two_body_e,(num_g,NUM_WALKERS),constants=[0],optimize='greedy')
expr_h2_o = contract_expression('ijG,GN->ijN',hamil_MF.two_body_o,(num_g,NUM_WALKERS),constants=[0],optimize='greedy')
expr_fb_e = contract_expression('Nri,irG->NG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_E,constants=[1],optimize='greedy')
expr_fb_o = contract_expression('Nri,irG->NG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_O,constants=[1],optimize='greedy')

expr_exch_new_new = contract_expression('Nri,jrG,Npj,ipG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_t,constants=[1,3],optimize='greedy')
expr_hart_new_new = contract_expression('Nri,irG,Npj,jpG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_t,constants=[1,3],optimize='greedy')



#exp_f = contract_expression('Nri,qirG->NqG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL,constants=[1])
#exp_ft = contract_expression('Nri,qirG->NqG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),ALPHA_FULL_T,constants=[1])
#expr_exch = contract_expression('Nri,jrG,Npj,ipG->N', (NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k), (num_electrons_up*num_k,num_orb*num_k,num_g),(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(num_electrons_up*num_k,num_orb*num_k,num_g))#,optimize='greedy')
#expr_exch_2 = contract_expression('Nri,qjrG,Npj,qipG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(num_k,num_electrons_up*num_k,num_orb*num_k,num_g),(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),(num_k,num_electrons_up*num_k,num_orb*num_k,num_g))


