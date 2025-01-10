import numpy as np
from mpi4py import MPI
from opt_einsum import contract_expression, contract


def main():
    NUM_WALKERS = 10
    num_k = 1
    num_orb = 8
    num_electrons_up = 4
    fsg = 0
    comm = MPI.COMM_WORLD
    PSI_T_up = np.eye(num_orb)[:,0:num_electrons_up]
    hamil_one_body = np.load("H1.npy")
    hamil_two_body = np.moveaxis(np.load("H2.npy"), 0, -1)
    mats_up = np.array(NUM_WALKERS * [PSI_T_up], dtype=np.complex64)
    weights = np.ones(NUM_WALKERS, dtype=np.complex64)

    alp = np.einsum('pi, prg -> irg',PSI_T_up , hamil_two_body)
    alp_t = np.einsum('pi, rpg -> irg', PSI_T_up, hamil_two_body.conj())
    alp_s = alp.astype(np.complex64)
    alp_s_t = alp_t.astype(np.complex64)
    expr_exch_single = contract_expression('Nri,jrG,Npj,ipG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_s,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_s_t,constants=[1,3],optimize='greedy')
    expr_hart_single = contract_expression('Nri,irG,Npj,jpG->N',(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_s,(NUM_WALKERS,num_orb*num_k,num_electrons_up*num_k),alp_s_t,constants=[1,3],optimize='greedy')
    E = measure_E_gs_single(PSI_T_up,weights,mats_up,hamil_one_body,NUM_WALKERS,num_k,num_electrons_up,fsg,expr_exch_single, expr_hart_single, comm)
    expected = 631.88947-2.8974954e-09j
    print(E, np.isclose(E, expected))
    np.random.seed(1887431)
    change = 0.05 * np.random.rand(*mats_up.shape)
    E = measure_E_gs_single(PSI_T_up,weights,mats_up + change,hamil_one_body,NUM_WALKERS,num_k,num_electrons_up,fsg,expr_exch_single, expr_hart_single, comm)
    expected = 631.8965-0.009482565j
    print(E, np.isclose(E, expected))

def measure_E_gs_single(trial,weights,walkers,h_1,NUM_WALKERS,num_k,num_electrons_up,fsg,expr_exch_single, expr_hart_single, comm):#,alpha_full,alpha_full_t):#,comm):
    thetas=[]
    for i in range(NUM_WALKERS):
        thetas.append(theta(trial, walkers[i]))
    thetas=np.array(thetas).astype(np.complex64)
    b=np.squeeze(np.dot(trial.T,h_1)).astype(np.complex64)
    e1=2*contract('iNi->N',np.tensordot(b, thetas,axes=((1,1)))) / num_k
    har_list  = 2* expr_hart_single(thetas,thetas) / num_k
    exch_list = (-expr_exch_single(thetas,thetas) - num_electrons_up * num_k *fsg)/num_k
    print(e1[0], har_list[0], exch_list[0])
    e_locs=e1+har_list+exch_list
    weights = weights.astype(np.complex64)
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
    return val_glb/sum_glb

def theta(trial, walker):
    return np.dot(walker, np.linalg.inv(overlap(trial,walker)))

def overlap(left_slater_det, right_slater_det):
    '''
    It computes the overlap between two Slater determinants.
    '''
    overlap_mat = np.dot(left_slater_det.transpose(),right_slater_det)
    #overlap_mat = np.array([right_slater_det[num_orb*k:num_orb*k+num_electrons_up] for k in range(num_k)]).reshape(num_k*num_electrons_up,num_k*num_electrons_up)
    return overlap_mat

if __name__ == "__main__":
    main()
