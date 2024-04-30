from afqmc_funcs_kpts_3 import *
#first_cpu = comm.Get_rank()==0
from time import time
import sys
from os.path import exists
log_file = 'log'
from scipy.linalg import expm
#sys.stdout = open(log_file, "w")

if first_cpu:
    print('###########################')
    print('###########################')
    print('###########################')
    print()
    print('    AA      FFFFF   QQQQQQ    MM      MM   CCCCCC')
    print('   A  A     F       Q    Q    M M    M M   C     ')
    print('  AAAAAA    FFFFF   Q    Q    M  M  M  M   C     ')
    print(' A      A   F       Q    Q    M   MM   M   C     ')
    print('A        A  F       QQQQQQQQ  M        M   CCCCCC')
    print()
    print('###########################')
    print('###########################')
    print('###########################')

    print()
    print('##########################')
    print('Input params:')
    print()
    print('system = ', system)
if SVD==True:
    if first_cpu:
        print('svd_trshd = ', svd_trshd)
if first_cpu:
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
    print('UPDATE_METHOD = ', UPDATE_METHOD)
    print('REORTHO_PERIODICITY = ', REORTHO_PERIODICITY)
    print('REBAL_PERIODICITY = ', REBAL_PERIODICITY)
    print('output_file = ', output_file)
    print()
    print('##########################')

    print()
    print('###########################')
    print('Rading Hamiltonian...')
hamil = HAMILTONIAN
h1 = np.array(read_datafile(input_file_one_body_hamil), order='C')
#h1 = np.zeros_like(h1)

hamil.two_body = np.array(read_datafile(input_file_two_body_hamil),dtype=np.complex64)
if first_cpu:
    print('Constructing Hermitian Conjugate')

h2_t = np.einsum('rpG->prG', hamil.two_body.conj())

if first_cpu:
    print('Hermitian Cojugate constructed Successfully')
    print()
    print('###########################')
    print()

    print()
    print('###########################')
    print()
    print('Checking Shapes...')
    print()
    print('hamil_two_body.shape -> ', hamil.two_body.shape)
    print('Number of orbitals = ', int(hamil.two_body.shape[0]/num_k))
    print('Number of AF fields = ', hamil.two_body.shape[2] )
    print()
    print('###########################')
    print()

    ###################################################
    #
    #   Read Q_list
    #
    ###################################################
    print('##########################')
    print()
    print ('Reading Q_list...')
ql = np.array(read_datafile(q_list),order='C').T
if first_cpu:
    print('Q_list read successfully')

###################################################
#
#  Generate minus_q
#
###################################################
m_q = gen_minus_q(ql)
if first_cpu:
    print('m_q = ', m_q)


###################################################
#
#  Reshape H1
#
###################################################
H1=np.zeros([num_orb*num_k,num_orb*num_k],dtype=np.complex64)
for i in range(0,num_k):
    H1[i*num_orb:(i+1)*num_orb,i*num_orb:(i+1)*num_orb] = h1[:,:,i]
for i in range(0,num_k):
    if first_cpu:
        print('H1 check?',np.allclose(H1[i*num_orb:(i+1)*num_orb,i*num_orb:(i+1)*num_orb] ,h1[:,:,i]))
hamil.one_body = H1

print('h1 -> ', hamil.one_body.itemsize)
print('h2 -> ', hamil.two_body.itemsize)
print('h2_t->', h2_t.itemsize)

if first_cpu:
    print()
    print('###########################')
    print('###########################')
    print('###########################')
    print()
    print('MM      MM  FFFFFF   SSSSSS  U     U  BBBBB ')
    print('M M    M M  F        SS      U     U  B   B ')
    print('M  M  M  M  FFFFFF     SS    U     U  BBBBBB')
    print('M   MM   M  F            SS  U     U  B    B')
    print('M        M  F        SSSSSS  UUUUUUU  BBBBBB')
    print()
    print('###########################')
    print('###########################')
    print('###########################')
    print()
    print('MF SUB in progress...')
hamil_MF = HAMILTONIAN_MF
#h2_af_MF_sub = A_af_MF_sub(PSI_T_up_0,PSI_T_up,hamil.two_body,ql)
#h2_af_MF_sub_new = A_af_MF_sub_new(PSI_T_up,hamil.two_body,ql)
h2_af_MF_sub = np.load('H2_af.npy')

hamil_MF.zero_body = H_0_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql)
H_zero = hamil_MF.zero_body
hamil_MF.one_body = H_1_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql,hamil.one_body)

'''
hamil_MF.two_body_e = gen_A_e_full(h2_af_MF_sub)
hamil_MF.two_body_o = gen_A_o_full(h2_af_MF_sub)
#print(np.round(hamil_MF.two_body_o[1,:,:,1],4))
if first_cpu:
    print('Checking the averages of AeQs and AoQs after mean-filed subtraction...')
for qc in range(1,num_k+1):
    if first_cpu:
        print('max(avg_Ae_Q) = ', max(avg_A_Q(PSI_T_up_0,PSI_T_up,hamil_MF.two_body_e,ql,qc)))
        print('min(avg_Ae_Q) = ', min(avg_A_Q(PSI_T_up_0,PSI_T_up,hamil_MF.two_body_e,ql,qc)))
        print('max(avg_Ao_Q) = ', max(avg_A_Q(PSI_T_up_0,PSI_T_up,hamil_MF.two_body_o,ql,qc)))
        print('min(avg_Ao_Q) = ', min(avg_A_Q(PSI_T_up_0,PSI_T_up,hamil_MF.two_body_o,ql,qc)))
'''
if first_cpu:
    print('MF SUB completed.')
    print()

    print()
    print('##########################')
    print('Self-energy calculation in progress...')
h_self = -contract('ijG,jkG->ik',hamil.two_body,h2_t)/2#/num_k
H_1_self = -D_TAU * (hamil_MF.one_body+h_self)
H1_self_exp = expm(H_1_self)
H1_self_half_exp = expm(H_1_self/2)
del(h_self)
if first_cpu:
    print('Self-energy calculation completed.')
    print()


    print('###########################')
    print('###########################')
    print('###########################')
    print()
    print('H      H  FFFFFF    EEEEEE  NN    N  EEEEEE  RRRRRR  GGGGGGG  Y   Y')
    print('H      H  F         E       NNN   N  E       R    R  G        Y   Y' )
    print('HHHHHHHH  FFFFFF    EEEEEE  N  N  N  EEEEEE  RRRRRR  G   GGG  YYYYY')
    print('H      H  F         E       N   NNN  E       R RR    G     G      Y')
    print('H      H  F         EEEEEE  N    NN  EEEEEE  R   RR  GGGGGGG  YYYYY')
    print()
    print('###########################')
    print('###########################')
    print('###########################')
walkers = WALKERS
#ALPHA_E = np.einsum('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_e)
#print("alpha_e: Memory size of numpy array in bytes:",ALPHA_E.size * ALPHA_E.itemsize/1e09)
#ALPHA_O = np.einsum('ip,prG->irG',PSI_T_up.T,hamil_MF.two_body_o)
#print("alpha_o: Memory size of numpy array in bytes:",ALPHA_O.size * ALPHA_O.itemsize/1e09)
#ALPHA_FULL = get_alpha_full(PSI_T_up,hamil.two_body,ql)
#print("alpha_full: Memory size of numpy array in bytes:",ALPHA_FULL.size * ALPHA_FULL.itemsize/1e09)
#print(ALPHA_FULL.shape)
#print(ALPHA_FULL.size)
#ALPHA_FULL_T = get_alpha_full_t(PSI_T_up,h2_t,ql,m_q)
#ALPHA_FULL_T = get_alpha_full_t(PSI_T_up,h2_t,ql,m_q)
#print("alpha_full_t: Memory size of numpy array in bytes:",ALPHA_FULL_T.size * ALPHA_FULL_T.itemsize/1e09)

#energy = measure_E_gs(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,hamil.two_body,h2_t,ql,m_q,ALPHA_FULL,ALPHA_FULL_T,comm)
energy_time_st = time()
energy = energy_new = measure_E_gs(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,0)#,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
print('energy time = ', time()-energy_time_st)

#t1=time()
#energy_test = measure_E_gs_new(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body)#,alp,alp_t)#,comm)
#print('energy test time = ', time()-t1)

#energy = 19.95523419043512*num_k+2*num_electrons_up*num_electrons_up*num_k*fsg*num_k 
if first_cpu:
    print('HF energy from new local energy routines = ', energy/num_k-2*num_electrons_up*num_electrons_up*num_k*fsg)

if HF_TEST==True:
    if first_cpu:
        print('###########################')
        print('###########################')
        print('###########################')
        print()
        print('H      H  FFFFFF   TTTTTTT  EEEEEE  SSSSSS  TTTTTTT')
        print('H      H  F           T     E       SS         T   ')
        print('HHHHHHHH  FFFFFF      T     EEEEEE    SS       T   ')
        print('H      H  F           T     E           SS     T   ')
        print('H      H  F           T     EEEEEE  SSSSSS     T   ')
        print()
        print('###########################')
        print('###########################')
        print('###########################')
        print()
        print(' tau                e_Hf                       d_e',)
        print('===================================================')

    HF_TEST_H_1_self = -HF_TEST_tau * (hamil_MF.one_body+h_self)
    for mul in range(0,HF_TEST_steps):
        for i in range(0,NUM_WALKERS):
            walkers.mats_up[i] = np.dot(propagator_fp(hamil_MF, HF_TEST_H_1_self,HF_TEST_tau,0),walkers.mats_up[i])
        val=np.array([])
        for walker_mat in walkers.mats_up:
            out = average_Hamil(hamil_MF, HF_TEST_H_1_self, PSI_T_up, walker_mat, NUM_WALKERS, HF_TEST_tau, 0, mul+2)
            val=np.append(val,out)
        res = np.mean(val)
        std_err = np.std(val)/np.sqrt(len(val)-1)
        txt = str(HF_TEST_tau*(mul+1)) + '\t' + str(res/num_k-2*num_k*num_electrons_up*num_electrons_up*fsg) + '\t' + str(std_err) + '\n'
        print (txt)

energy_new = energy
e_hf=energy
e_hf_1 = e_hf
j = 1
wr_name='WALKERS.npy'
wt_name='WEIGHT.npy'
file_exists = exists(wr_name)
if file_exists:
    walkers.mats_up=np.load(wr_name)
    walkers.weights=np.load(wt_name)
    energy_new = measure_E_gs(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,0)
    if first_cpu:
        print(energy_new/num_k-2*num_k*num_electrons_up*num_electrons_up*fsg)
        #print('HF energy from test local energy routines = ', energy_test/num_k-2*num_electrons_up*num_electrons_up*num_k*fsg)
e_hf=energy_new
if file_exists:
    file_out = open(output_file,"a")
else:
    file_out = open(output_file,"w")

if first_cpu:
    print()
    print('###########################')
    print()
    print('AFQMC simulation...')
    print()
num_rare = 0
start_time = time()
update_method = set_update_method(UPDATE_METHOD)
while (j<NUM_STEPS+1):
    old_walkers_up = walkers.mats_up
    old_weights = walkers.weights
    update_time_st=time()
    walkers.mats_up,walkers.weights = update_walker(PSI_T_up_0,PSI_T_up,walkers.mats_up,walkers.weights,ql,H_zero,H_1_self,D_TAU,e_hf,update_method,H1_self_exp,H1_self_half_exp)
    #print('walkers.mats_up,walkers.weights ', walkers.mats_up.itemsize,walkers.weights.itemsize)
    #print('walkers.weights = ', walkers.weights)
    #print('np.sum(walkers.weights) = ', np.sum(walkers.weights))
    walkers.weights=walkers.weights/np.sum(walkers.weights)
    #print('walkers.weights = ', walkers.weights)
    #print('np.sum(walkers.weights) = ', np.sum(walkers.weights))
    print('update time = ', time()-update_time_st)
    #for l in range (0, NUM_WALKERS):
        #walkers.mats_up[l] = reortho_qr(walkers.mats_up[l])
    if j%CHECK_PERIODICITY==0:
        np.save(wr_name,walkers.mats_up)
        np.save(wt_name,walkers.weights)

    #walkers.mats_up,walkers.weights = update_hyb(PSI_T_up_0,PSI_T_up,walkers.mats_up,walkers.weights,ql,hamil_MF,H_1_self,D_TAU,e_hf,hamil,h2_t,ALPHA_E,ALPHA_O)
    #walkers.mats_up,walkers.weights = update_fp(PSI_T_up_0,PSI_T_up,walkers.mats_up,walkers.weights,ql,hamil_MF,H_1_self,D_TAU,e_hf,hamil,h2_t,ALPHA_E,ALPHA_O)
    #energy_new = measure_E_gs(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,hamil.two_body,h2_t,ql,m_q,ALPHA_FULL,ALPHA_FULL_T,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
    if j%SAMP_FREQ==0:
        energy_time_st = time()
        energy_new = measure_E_gs(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,e_hf)#,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
        print('energy time = ', time()-energy_time_st)
        #t1=time()
        #energy_new_test = measure_E_gs_new(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body)#alp,alp_t)#,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
        #print('energy test time = ', time()-t1)
        #print('???',energy_new,energy_new_test)

    if True: #abs(energy.imag)<abs(trsh_imag) and abs(energy.real-e_hf.real)<abs(trsh_real) :#ratio*(energy.real+en_const))):# and (abs((energy.real-e_hf.real)/(en_const+e_hf.real))<MAX_ACC_VAL):
        e_hf=e_hf*j+energy_new
        e_hf = e_hf/(j+1)
        if first_cpu:
            print(j*D_TAU, e_hf/num_k-2*num_k*num_electrons_up*num_electrons_up*fsg)
            txt = str(j*D_TAU) + '\t' + str(energy_new.real/num_k-2*num_k*num_electrons_up*num_electrons_up*fsg) + '\t' + str(energy_new.imag) + '\n'
            file_out.write(txt)
            print()
        #print('NUM_WALKERS = ', NUM_WALKERS)
        if j%REORTHO_PERIODICITY == 0:
            for i in range (0, NUM_WALKERS):
                walkers.mats_up[i] = reortho_qr(walkers.mats_up[i])
            #print('NUM_WALKERS = ', NUM_WALKERS)
        if REBAL_PERIODICITY!=0 and j%REBAL_PERIODICITY==0:
            rebalanced_weights_indices = rebalance_comb(walkers.weights)
            walkers.mats_up = walkers.mats_up[rebalanced_weights_indices]
            walkers.weights = init_walkers_weights(NUM_WALKERS)
        j+=1
    else:
        if first_cpu: 
            print('RRRRRR        A        RRRRRR   EEEEEE    EEEEEE V           V  EEEEEE  N     N  TTTTTTTTT')
            print('R    R       A A       R    R   E         E       V         V   E       NN    N      T')
            print('RRRRRR      A   A      RRRRRR   E         E        V       V    E       N N   N      T')
            print('R R        AAAAAAA     R R      EEEEEE    EEEEEE    V     V     EEEEEE  N  N  N      T')
            print('R  R      A       A    R  R     E         E          V   V      E       N   N N      T')
            print('R   R    A         A   R   R    E         E           V V       E       N    NN      T')
            print('R    R  A           A  R    R   EEEEEE    EEEEEEE      V        EEEEEE  N     N      T')
        num_rare+=1
        walkers.mats_up = old_walkers_up
        walkers.weights = old_weights
        for i in range (0, NUM_WALKERS):
            walkers.mats_up[i] = reortho_qr(walkers.mats_up[i])
        rebalanced_weights_indices = rebalance_comb(walkers.weights)
        walkers.mats_up = walkers.mats_up[rebalanced_weights_indices]
        walkers.weights = init_walkers_weights(NUM_WALKERS)
file_out.close()
if first_cpu:
    print('number of rare events = ',num_rare)
    print('total runtime = ', -start_time+time())
    print('Please see', output_file)
    print()
    print('##########################')
    print()
'''
file_out = open(output_file,"r")
data1=np.loadtxt(file_out)
data = data1[:,[1]]
st = int(len(data)*EQUILIBRATION)
en = len(data)
a = blockAverage(data[st:en])
if first_cpu:
    print()
    print('Calculating ground state energy...')
    print()
    print ('E_gs_afqmc = ', np.mean(data[st:en]), '+-', np.sqrt(max(a[1])))
file_out.close()
if first_cpu:
    print()
'''

