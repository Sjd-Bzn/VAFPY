import numpy as np
from scipy.sparse import block_diag
from opt_einsum import contract_expression

num_k = 8
num_band = 8
num_elec = 4

L = np.load("H2.npy")
print('H2 = ', L)
k1_list = np.load("k1_list.npy") - 1
num_g = len(L) // num_k



mask = np.array(8 * (4 * [True] + 4 * [False]))
print("Mask = ", mask)
mask1 = np.array(4 * [True] + 4 * [False])
L_occ = L[:, mask1][:, :, mask]
print('L_occ = ', L_occ.shape)

Ex1 = np.einsum("gij,gij->", L_occ, L_occ.conj()) / num_k
print('Ex1 = ', Ex1)

hf_det = block_diag([np.eye(num_band, num_elec)]).toarray()
hf_det_1 = block_diag(num_k * [np.eye(num_band, num_elec)]).toarray()
hf_det_2 = block_diag([np.eye(num_k * num_band, num_band)]).toarray()
alpha = np.einsum("ij,gmi->jmg", hf_det_1, L)
beta = np.einsum("ij,gmi->jmg", hf_det_1, L.conj())
print("alpha shape", alpha.shape)
print("beta shape", beta.shape)

expr_exch = contract_expression(
    "jk,rjG,pk,rpG->",
    hf_det.shape,
    alpha,
    hf_det.shape,
    beta,
    constants=[1, 3],
    optimize="greedy",
)
Ex2 = expr_exch(hf_det, hf_det) / num_k
print(f"{Ex2=}")

E_har_1 = 0
first = 0
second = 0
for i in range (num_k):
    L_k = L_occ[:num_g,: ,i*num_elec:(i+1)*num_elec]
    first += np.einsum("gii->g", L_k)
    second += np.einsum("gjj->g", L_k.conj())
#    E_har_1 += 2* np.einsum("gii,gjj->", L_k, L_k.conj()) / num_k
E_har_1 = 2 * np.einsum("g,g->", first, second) / num_k
print("first shape", first.shape)



#L_new = np.zeros((L_occ.shape[0], L_occ.shape[2] , L_occ.shape[3]*8))
#
## Loop over the second dimension of L_occ[2], processing every group of 8 elements
#for i in range( 8):  # Assumes the dimension is divisible by 8
#    L_new += np.sum(L_occ[2, :,i*8:(i+1)*8], axis=0)
#
#L_new = L_new.reshape(L_new.shape[0], L_new.shape[1]*8, L_new.shape[2]/8)
#E_har_1 = 2* np.einsum("gii,gjj->", L_new, L_new.conj()) / num_k
print('E_har_1 = ', E_har_1)

alpha_full = np.zeros([num_elec*num_k, num_band*num_k, len(L) ],dtype=np.complex128)
beta_full = np.zeros([num_elec*num_k, num_band*num_k, len(L) ],dtype=np.complex128)
for j in range(num_k):
    for i in range (num_k):
        k = k1_list[i,j]
        alpha_full[i*num_elec:(i+1)*num_elec,k*num_band:(k+1)*num_band,j*num_g:(j+1)*num_g] \
            = alpha[i*num_elec:(i+1)*num_elec,:,j*num_g:(j+1)*num_g]
        beta_full[i*num_elec:(i+1)*num_elec,k*num_band:(k+1)*num_band,j*num_g:(j+1)*num_g] \
            = beta[i*num_elec:(i+1)*num_elec,:,j*num_g:(j+1)*num_g]

print("alpha_full shape", alpha_full.shape)
print("beta_full shape", beta_full.shape)


expr_har = contract_expression(
    "ri,irG,pj,jpG->",
    hf_det_1.shape,
    alpha_full,
    hf_det_1.shape,
    beta_full,
    constants=[1, 3],
    optimize="greedy",
)
E_har_2 = 2*expr_har(hf_det_1, hf_det_1) / num_k
print(f"{E_har_2=}")


expr_exch_2 = contract_expression(
    "jk,rjG,pk,rpG->",
    hf_det_1.shape,
    alpha_full,
    hf_det_1.shape,
    beta_full,
    constants=[1, 3],
    optimize="greedy",
)
Ex3 = expr_exch_2(hf_det_1, hf_det_1) / num_k
print(f"{Ex3=}")



