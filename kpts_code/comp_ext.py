import numpy as np
import time

def get_q_list(q_list,q_selected):
    '''
    It returns q_list for a specicific momentum q_selected.
    '''
    return q_list[q_list[:,2]==q_selected]

def get_k1s_k2s(q_list,q_selected):
    '''
    It retuirns a list of tuples (k1s,k2s) corrsponding to q_selected.
    '''
    return list(zip(get_q_list(q_list,q_selected)[:,0],get_q_list(q_list,q_selected)[:,1]))

def ext_A_Q(aq,q_list,q_selected):
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    result = np.zeros([nb*nk,nb*nk,nw],np.complex128)
    index=0
    for K1,K2 in K1s_K2s:
        result[(K1-1)*nb:K1*nb,(K2-1)*nb:K2*nb]=aq[index]
        index+=1
    return result

def ext_A_Q_new2(result,aq,q_list,q_selected):
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    index=0
    for K1,K2 in K1s_K2s:
        result[(K1-1)*nb:K1*nb,(K2-1)*nb:K2*nb] = aq[index]
        index+=1

def ext_A_Q_new(result,aq,q_list,q_selected):
    K1s_K2s = get_k1s_k2s(q_list,q_selected)
    index=0
    for K1,K2 in K1s_K2s:
        result[(K1-1)*nb:K1*nb,(K2-1)*nb:K2*nb]+=aq[index]
        index+=1



nk = 27             # number of k points
nb = 8              # numbr of bands
nw = 512             # number of walkers
ng = 193            # number G vectors for the prim-cell H2
NG = 5169           # number of G vectors for the supercell H2
                    # ng ~ NG/nk
ql=np.load('Q_list_3k.npy').T


'''
nk = 64
nb = 8
nw = 256
ng = 161
NG = 10291

ql=np.load('Q_list_4k.npy').T
'''
'''
nk = 27             # number of k points
nb = 16              # numbr of bands
nw = 256             # number of walkers
ng = 275            # number G vectors for the prim cell H2
NG = 7390           # number of G vectors for the sc H2
                    # ng ~ NG/nk
ql=np.load('Q_list_3k.npy').T
'''


'''
nk = 8              # number of k points
nb = 8              # numbr of bands
nw = 512             # number of walkers
ng = 184            # number G vectors for the prim cell H2
NG = 1414           # number of G vectors for the sc H2
                    # ng ~ NG/nk
ql=np.load('Q_list_2k.npy').T
'''


print()
h2_sc = np.random.random([nk*nb,nk*nb,NG]) + 1j*np.random.random([nk*nb,nk*nb,NG])  # supercell H2
print('h2_sc.shape = ', h2_sc.shape)

h2_prim = np.random.random([nk,nk,nb,nb,ng]) + 1j*np.random.random([nk,nk,nb,nb,ng])    # prim-cell H2
print('h2_prim.shape = ', h2_prim.shape)
print()

h2_ext = np.zeros([nb*nk,nb*nk,nw],np.complex128)   # to store effective hamiltonian 


##################################################
########### h2_eff prim approach 1 ###############
##################################################
x=np.random.random([ng,nw,nk])+1j*np.random.random([ng,nw,nk])    # random numbers
'''
t1 = time.time()
for num_calc in range(0,10):
    for q in range(0,nk):
        h2_ext += ext_A_Q(h2_prim[q]@x[:,:,q],ql,q+1)
print('time prim 1 = ', time.time()-t1)
print()
'''
##################################################
########### h2_eff prim approach 2 ###############
##################################################
y=np.moveaxis(x,-1,0).copy()       # reshaped random numbers
h2_alt = np.zeros_like(h2_ext)     # to store effective hamiltonian

t1 = time.time()
for q in range(0,nk):
    h2_alt += ext_A_Q(h2_prim[q] @ y[q], ql, q+1)
print('time prim 2 = ', time.time()-t1)

#print('???', np.allclose(h2_alt,h2_ext))

print()

##################################################
########### h2_eff prim approach 3 ###############
##################################################


t1 = time.time()
result = np.zeros([nb*nk,nb*nk,nw],np.complex128)
for q in range(0,nk):
    ext_A_Q_new(result, h2_prim[q] @ y[q], ql, q+1)
print('time prim 3 = ', time.time()-t1)



print('???', np.allclose(h2_alt,result))

##################################################
########### h2_eff prim approach 4 ###############
##################################################

result2 = np.zeros([nb*nk,nb*nk,nw],np.complex128)
for q in range(0,nk):
    aq = (h2_prim[q].reshape(nk*nb*nb,ng) @ y[q]).reshape(nk,nb,nb,nw)
    ext_A_Q_new2(result2, aq, ql, q+1)
print('time prim 4 = ', time.time()-t1)
print()

print('???', np.allclose(h2_alt,result))
print('???', np.allclose(result2,result))

####################################
############h2_eff sc###############
####################################
x=np.random.random([NG,nw])+1j*np.random.random([NG,nw])
t1 = time.time()
for num_calc in range(0,1):
    h2_ext = h2_sc@x
print('time sc = ', time.time()-t1)
print()

