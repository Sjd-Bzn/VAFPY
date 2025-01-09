import numpy as np
from scipy.sparse import block_diag
from opt_einsum import contract_expression
from scipy.linalg import expm
from cmath import phase


num_orb = 8
num_k = 8
num_band = 8
num_elec = 4
d_tau = 0.0001
NUM_WALKERS = 2
H1 = np.load("H1.npy")
L = np.load("H2.npy")
print('H2 = ', L)

num_g = L.shape[0]
NG = num_g

mask = np.array(8 * (4 * [True] + 4 * [False]))
print("Mask = ", mask)
mask_1 = np.array(4 *[True] + 4 * [False])
L_occ = L[:, mask][:, :, mask]
print('L_occ = ', L_occ)

Ex1 = np.einsum("gij,gij->", L_occ, L_occ.conj()) / num_k
print('Ex1 = ', Ex1)
H1_occ = H1[mask_1][:,mask_1]
print("H1_occ", H1_occ.shape)
def theta(trial, walker):

    return np.dot(walker, np.linalg.inv(overlap(trial,walker)))
def overlap(left_slater_det, right_slater_det):
    overlap_mat = np.array([right_slater_det[num_orb*k:num_orb*k+num_elec] for k in range(num_k)]).reshape(num_k*num_elec,num_k*num_elec)
    return overlap_mat


hf_det = block_diag(num_k * [np.eye(num_band, num_elec)]).toarray()
alpha = np.einsum("ni,gnm->img", hf_det, L)


walker_weight = np.ones(NUM_WALKERS, dtype=np.complex128)
walker_mat = np.array(NUM_WALKERS * [hf_det], dtype=np.complex128)
print("walker_mat", walker_mat.shape)

def reshape_H1(H1, num_k, num_orb):
    '''
    Reshape H1 (H1.npy shape is num_orb, num_orb, num_k   
    we implement the k_points to the H1)
    '''
    h1=np.zeros([num_band*num_k,num_band*num_k],dtype=np.complex128)
    for i in range(0,num_k):
        h1[i*num_band:(i+1)*num_band,i*num_band:(i+1)*num_band] = H1[:,:,i]
    for i in range(0,num_k):
        print('H1 check?',np.allclose(h1[i*num_band:(i+1)*num_band,i*num_band:(i+1)*num_band] ,H1[:,:,i]))
    return h1
H_1 = reshape_H1(H1, num_k, num_elec)
print("H_1", H_1.shape)

def exp_Taylor(mat):
    OUT = np.eye(num_band*num_k,dtype = 'complex128')
    C = np.eye(num_band*num_k)
    for i in range (0,5):
      C = mat@C/(i+1)
      OUT += C
    return OUT

def update_hyb(trial,walker_mat,walker_weight,h_1,d_tau):
    SQRT_DTAU = np.sqrt(d_tau)
    #walker_weight = np.ones(NUM_WALKERS, dtype=np.complex128)
    #walker_mat = np.array(NUM_WALKERS * [hf_det], dtype=np.complex128) 
    new_walker_mat = np.zeros_like(walker_mat)
    new_walker_weight = np.zeros_like(walker_weight)
    x = np.random.randn(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS)
    

    theta_mat = []
    for i in range(NUM_WALKERS):
        theta_mat.append(theta(L, walker_mat[i]))     ### maybe L_occ
    theta_mat=np.array(theta_mat)
    print("theta_mat", theta_mat.shape)

    expr_fb = contract_expression('Nri,irG->NG',(NUM_WALKERS,num_orb*num_k,num_elec*num_k),alpha,constants=[1],optimize='greedy')
    fb = -2j*SQRT_DTAU*expr_fb(theta_mat)
    print("fb", fb.shape)
   
    expr_h2 = contract_expression('Gij,GN->ijN',L,(num_g,NUM_WALKERS),constants=[0],optimize='greedy')
    h_2 = expr_h2(x - fb.T)
    print("h_2",h_2.shape)
    for i in range(0,NUM_WALKERS):
        h=-d_tau*h_1+SQRT_DTAU*1j*h_2[:,:,i]
        print("h",h.shape)
#### S2 
        prop_S2 = expm(-d_tau * h_1 / 2) @ exp_Taylor(SQRT_DTAU * 1j * h_2[:, :, i]) @ expm(-d_tau * h_1 / 2)
        print("prop_S2", prop_S2.shape)
        new_walker_mat[i] = prop_S2 @ walker_mat[i]
##### NEW_WEIGHT
        ovrlap_ratio = np.linalg.det(overlap(L,new_walker_mat[i]))**2 / np.linalg.det(overlap(L,walker_mat[i]))**2
        Phasee = phase(ovrlap_ratio)
        new_walker_weight[i] = abs(ovrlap_ratio*(np.exp( np.dot(x[:,i],fb[i])-np.dot(fb[i],fb[i]/2))))* max(0,np.cos(Phasee))*walker_weight[i]
    return new_walker_mat, new_walker_weight

new_walkers,walker_weights = update_hyb(L,walker_mat,walker_weight,H_1,d_tau)
print("walkers", new_walkers,walker_weights)
walker_weights=walker_weights/np.sum(walker_weights)

hf_det = block_diag(num_k * [np.eye(num_band, num_elec)]).toarray()
alpha = np.einsum("ni,gnm->img", hf_det, L)
beta = np.einsum("ni,gmn->img", hf_det, L.conj())
print('alpha shape', alpha.shape)
print('beta shape', beta.shape)
print("hf det", hf_det.shape)

thetas=[]
for i in range(NUM_WALKERS):
    thetas.append(theta(L, new_walkers[i]))
thetas=np.array(thetas)
print("thetas", thetas.shape)    

w = (NUM_WALKERS,num_orb*num_k,num_elec*num_k)
expr_exch = contract_expression(
    "Nri,jrG,Npj,ipG->N",
    w,
    alpha,
    w,
    beta,
    constants=[1, 3],
    optimize="greedy",
)
Ex2 = expr_exch(thetas, thetas) / num_k
print(f"{Ex2=}")


mask = np.array(8 * (4 * [True] + 4 * [False]))
#print("Mask = ", mask)
#
L_occ = L[:, mask][:, :, mask]
A = np.einsum("gii->g", L_occ[:num_g])
B = np.einsum("gjj->g", L_occ[:num_g].conj())
E_har_3 = 2* np.einsum("g,g->", A, B) / num_k

print('E_har_3 = ', E_har_3)
print("A", A)
E_har_1 = 2* np.einsum("gii,gjj->", L_occ, L_occ.conj()) / num_k
print('E_har_1 = ', E_har_1)




expr_har = contract_expression(
    "ri,irG,pj,jpG->",
    hf_det.shape,
    alpha,
    hf_det.shape,
    beta,
    constants=[1, 3],
    optimize="greedy",
)
E_har_2 = 2*expr_har(hf_det, hf_det) / num_k
print(f"{E_har_2=}")






