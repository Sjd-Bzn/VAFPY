import numpy as np
from scipy.sparse import block_diag
from opt_einsum import contract_expression

num_k = 8
num_band = 8
num_elec = 4
D_TAU = 0.0001


L = np.load("H2.npy")
print('H2 = ', L)

num_g = L.shape[0]//num_k
NG = num_g

mask = np.array(8 * (4 * [True] + 4 * [False]))
print("Mask = ", mask)

L_occ = L[:, mask][:, :, mask]
print('L_occ = ', L_occ)

Ex1 = np.einsum("gij,gij->", L_occ, L_occ.conj()) / num_k
print('Ex1 = ', Ex1)


def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial,walker)))
def overlap(left_slater_det, right_slater_det):
    overlap_mat = np.array([right_slater_det[num_orb*k:num_orb*k+num_electrons_up] for k in range(num_k)]).reshape(num_k*num_electrons_up,num_k*num_electrons_up)
    return overlap_mat


hf_det = block_diag(num_k * [np.eye(num_band, num_elec)]).toarray()
alpha = np.einsum("ni,gnm->img", hf_det, L)



def update_hyb(trial,walker_mat,walker_weight,h_1,d_tau):
    SQRT_DTAU = np.sqrt(d_tau)
    walker_weight = np.one(NUM_WALKERS, dtype=np.complex128)
    walker_mat = np.array(NUM_WALKERS * [hf_det], dtype=np.complex128) 
    new_walker_mat = np.zeros_like(walker_mat)
    new_walker_weight = np.zeros_like(walker_weight)
    x = np.random.randn(NG*NUM_WALKERS).reshape(NG,NUM_WALKERS)
    

    theta_mat = []
    for i in range(NUM_WALKERS):
        theta_mat.append(theta(L, walker_mat[i]))     ### maybe L_occ
    theta_mat=np.array(theta_mat)

    expr_fb = contract_expression('Nri,irG->NG',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alpha,constants=[1],optimize='greedy')
    fb = -2j*SQRT_DTAU*expr_fb(theta_mat)
    
    expr_h2 = contract_expression('Gij,GN->ijN',L_occ,(num_g,NUM_WALKERS),constants=[0],optimize='greedy')
    h_2 = expr_h2(x - fb.T)
    for i in range(0,NUM_WALKERS):
        h=-D_TAU*h_1+SQRT_DTAU*1j*h_2[:,:,i]
#### S2 
        prop_S2 = expm(-d_tau * h_1 / 2) @ exp_Taylor(SQRT_DTAU * 1j * h_2[:, :, i]) @ expm(-d_tau * h_1 / 2)
        new_walker_mat[i] = prop_S2 @ walker_mat[i]
##### NEW_WEIGHT
        ovrlap_ratio = np.linalg.det(overlap(L,new_walker_mat[i]))**2 / np.linalg.det(overlap(L,walker_mat[i]))**2
        Phasee = phase(ovrlap_ratio)
        new_walker_weight[i] = abs(ovrlap_ratio*(np.exp( np.dot(x[:,i],fb[i])-np.dot(fb[i],fb[i]/2))))* max(0,np.cos(Phasee))*walker_weight[i]
    return new_walker_mat, new_walker_weight




hf_det = block_diag(num_k * [np.eye(num_band, num_elec)]).toarray()
alpha = np.einsum("ni,gnm->img", hf_det, L)
beta = np.einsum("ni,gmn->img", hf_det, L.conj())
print('alpha shape', alpha.shape)
print('beta shape', beta.shape)
print("hf det", hf_det.shape)

expr_exch = contract_expression(
    "ri,jrG,pj,ipG->",
    hf_det.shape,
    alpha,
    hf_det.shape,
    beta,
    constants=[1, 3],
    optimize="greedy",
)
Ex2 = expr_exch(hf_det, hf_det) / num_k
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






