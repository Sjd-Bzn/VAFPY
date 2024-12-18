from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() != 0:
    print = lambda *arg, **kwargs: None

from afqmc_funcs_kpts import *
#first_cpu = comm.Get_rank()==0
#import time
from time import time
import sys
from os.path import exists
log_file = 'log'
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

#np.random.seed(15462)

hamil = HAMILTONIAN
###########################################

####### reshape the H1.npy and H2.npy ######

############################################
### H1 double
H1 = np.array(read_datafile(input_file_one_body_hamil))
#print("H1 type", H1.dtype)
h1 = reshape_H1(H1, num_k, num_orb)
#h1 = h1.astype(np.complex64)
hamil.one_body = h1
H2 = np.array(read_datafile(input_file_two_body_hamil))
h2 = H2#np.moveaxis(H2, 0, -1) 
hamil.two_body = h2
#print("H2 type", H2.dtype)

#### H2 dager
h2_t = np.einsum('prG->rpG', hamil.two_body.conj())
print("H2 t type", h2_t.dtype)



if first_cpu:
    print('Hamiltonian constructed Successfully')
    print()
    print('###########################')

    print()
    print('###########################')
    print()
    print('Checking Shapes...')
    print()
    print('H1.npy and H2.npy shape')

    print('H1.npy shape -> ', H1.shape)
    print('H2.npy shape -> ', H2.shape)

    print( 'After reshaping ')
    print()
    print('h1 shape -> ', hamil.one_body.shape)
    print('h2 shape -> ', hamil.two_body.shape)
    print('h1 size in bytes -> ', hamil.one_body.itemsize)
    print('h2 size in bytes -> ', hamil.two_body.itemsize)
    print('h2_t size in bytes ->', h2_t.itemsize)
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
#ql = np.array(read_datafile(q_list),order='C').T
ql = []
for i in range(1,num_k+1):
    for j in range(1,num_k+1):
        for k in range(1,num_k+1):
            if abs(i-j)==k-1:
                ql.append([i,j,k])
ql = np.array(ql)



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
h2_af_MF_sub = A_af_MF_sub(PSI_T_up_0,PSI_T_up,hamil.two_body,ql)
#hamil_MF.two_body = A_af_MF_sub(PSI_T_up_0,PSI_T_up,hamil.two_body,ql)

#h2_af_MF_sub_new = A_af_MF_sub_new(PSI_T_up,hamil.two_body,ql)
#h2_af_MF_sub = np.load('H2_af.npy')
#h2_af_MF_sub_new_t = np.einsum('rpG->prG', h2_af_MF_sub_new.conj())

L_0 = mean_field(hamil.two_body, num_electrons_up, num_orb)


H_zero= np.einsum("g, g->", L_0, L_0 )/2/2/num_k
print("H_0", H_zero)
#hamil_MF.zero_body = # H_0_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql)
hamil_MF.zero_body= H_zero
print("H_zero", H_zero)


hamil_MF.one_body = H_1_mf(PSI_T_up_0,PSI_T_up,hamil.two_body,h2_t,ql,hamil.one_body)
#H1_exp = expm(H_1_self)
#H1_self_half_exp = expm(H_1_self/2)


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
#one_body_energy = E_one(PSI_T_up, walkers.weights, walkers.mats_up, hamil.one_body)/num_k 
#hartree = Hartree(PSI_T_up, walkers.weights, walkers.mats_up)/num_k
#exchange = (Exchange(PSI_T_up, walkers.weights, walkers.mats_up)/num_k)
#energy = energy_new 
d = measure_E_gs_double(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,0)#,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
energy = energy_new_d = d[0]
one_body_energy = E_one(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body) 
hartree = Hartree(PSI_T_up,walkers.weights,walkers.mats_up)
exchange = Exchange(PSI_T_up,walkers.weights,walkers.mats_up)
E1_vasp = EBANDS - 2 * hartree - 2* exchange




print('energy time = ', time()-energy_time_st)



#t1=time()
#energy_test = measure_E_gs_new(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body)#,alp,alp_t)#,comm)
#print('energy test time = ', time()-t1)

#energy = 19.95523419043512*num_k+2*num_electrons_up*num_electrons_up*num_k*fsg*num_k 
if first_cpu:
    print('H1 Energy from AFQMC = ', one_body_energy)
    print('H1 Energy from Vasp  = ', E1_vasp)
    print('Hartree Energy       = ', hartree)
    print('Exchange Energy      = ', exchange)
    print('HF energy from AFQMC = ', energy)
  #  print('HF energy from new local energy routines = ', energy_2/num_k-2*num_electrons_up*num_electrons_up*num_k*fsg)


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
    HF_TEST_H_1_self =-HF_TEST_tau * (h_self +hamil_MF.one_body )
    for mul in range(0,HF_TEST_steps):
        val=np.array([])
        for i in range(0,NUM_WALKERS):
        #    walkers.mats_up[i] = np.dot(propagator_fp(hamil_MF, HF_TEST_H_1_self,HF_TEST_tau,0),walkers.mats_up[i])
        #for walker_mat in walkers.mats_up:
            #out = average_Hamil(hamil_MF, hamil_MF.one_body, PSI_T_up, walkers.mats_up[i], NUM_WALKERS, HF_TEST_tau, 0, 1)
            out = average_Hamil(hamil_MF, HF_TEST_H_1_self, PSI_T_up, walkers.mats_up[i], NUM_WALKERS, HF_TEST_tau, 0, 1)
            val=np.append(val,out)
        res = np.mean(val)
        std_err = np.std(val)/np.sqrt(len(val)-1)
        txt = str(HF_TEST_tau*(mul+1)) + '\t' + str(res/num_k-2*num_k*num_electrons_up*num_electrons_up*fsg) + '\t' + str(std_err) + '\n'
        print (txt)

j = 1
wr_name='WALKERS.npy'
wt_name='WEIGHT.npy'
file_exists = exists(wr_name)
if file_exists:
    walkers.mats_up=np.load(wr_name)
    walkers.weights=np.load(wt_name)
    d = measure_E_gs_double(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,0)#,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
    energy_new = d[0]
    if first_cpu:
        print(energy_new)#-2*num_k*num_electrons_up*num_electrons_up*fsg)
        #print('HF energy from test local energy routines = ', energy_test/num_k-2*num_electrons_up*num_electrons_up*num_k*fsg)
#energy_new = energy
e_hf_d = energy_new_d
v_d = d[1]
w_d = d[5]
exch_d = d[2]
har_d = d[3]
one_d = d[4]

s = measure_E_gs_double(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,0)#,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
energy_new_s = s[0]
e_hf_s = energy_new_s
v_s = s[1]
w_s = s[5]
exch_s = s[2]
har_s= s[3]
one_s= s[4]




time_file = "times.txt"
if file_exists:
    t = open(time_file,"a")
else:
    t = open(time_file,"w")

profile_file = "profile.txt"
if file_exists:
    p = open(profile_file,"a")
else:
    p = open(profile_file,"w")

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
cumulative_time_energy_d = 0.0
time_log_energy_d = []
cumulative_time_energy_s = 0.0
time_log_energy_s = []
cumulative_v_differ = 0.0
log_v_differ = []
cumulative_exch_differ = 0.0
log_exch_differ = []
cumulative_har_differ = 0.0
log_har_differ = []
cumulative_one_differ = 0.0
log_one_differ = []


start_time = time()
#update_method = set_update_method(UPDATE_METHOD)
while (j<NUM_STEPS+1):
    s_update = time()
    walkers.mats_up,walkers.weights = update_hyb(PSI_T_up_0, PSI_T_up,walkers.mats_up,walkers.weights,ql,0,hamil.one_body,D_TAU,0,H1_self_half_exp,propagator)
    #time_update = time() - s_update
    #cumulative_update_time += time_update
    #time_log_update.append(time_update)  # Log the time per step

    # Write the current step time and cumulative time to file
    #t.write(f"{j}\t{time_update:.6f}\t{cumulative_update_time:.6f}\t")

  
    if j%CHECK_PERIODICITY==0:
        np.save(wr_name,walkers.mats_up)
        np.save(wt_name,walkers.weights)

    #walkers.mats_up,walkers.weights = update_hyb(PSI_T_up_0,PSI_T_up,walkers.mats_up,walkers.weights,ql,hamil_MF,H_1_self,D_TAU,e_hf,hamil,h2_t,ALPHA_E,ALPHA_O)
    #walkers.mats_up,walkers.weights = update_fp(PSI_T_up_0,PSI_T_up,walkers.mats_up,walkers.weights,ql,hamil_MF,H_1_self,D_TAU,e_hf,hamil,h2_t,ALPHA_E,ALPHA_O)
    #energy_new = measure_E_gs(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,hamil.two_body,h2_t,ql,m_q,ALPHA_FULL,ALPHA_FULL_T,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
    if j%SAMP_FREQ==0:
        t_d=time()
        d=  measure_E_gs_double(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,e_hf_d)#,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
        energy_d_time = time() - t_d
        cumulative_time_energy_d += energy_d_time
        time_log_energy_d.append(energy_d_time)  # Log the time per step

        # Write the current step time and cumulative time to file
        t.write(f"{j}\t{energy_d_time:.6f}\t{cumulative_time_energy_d:.6f}\t")
        #print(f"Step: {j}, Time: {time_double:.6f} s, Cumulative Time: {cumulative_time_double:.6f} s")        


        energy_new_d= d[0]
        v_d = d[1]
        w_d = d[5]
        exch_d = d[2]
        har_d = d[3]
        one_d = d[4]
        #print('energy test time = ', time()-t1)
        
        t_s=time()
        s=  measure_E_gs_single(PSI_T_up,walkers.weights,walkers.mats_up,hamil.one_body,e_hf_s)#,comm)#-2*num_electrons_up*num_electrons_up*num_k*fsg
        energy_s_time = time() - t_s
        cumulative_time_energy_s += energy_s_time
        time_log_energy_s.append(energy_s_time)  # Log the time per step

        # Write the current step time and cumulative time to file
        t.write(f"{energy_s_time:.6f}\t{cumulative_time_energy_s:.6f}\n")
        #print(f"Step: {j}, Time: {time_double:.6f} s, Cumulative Time: {cumulative_time_double:.6f} s")        


        energy_new_s= s[0]
        v_s = s[1]
        w_s = s[5]
        exch_s = s[2]
        har_s = s[3]
        one_s = s[4]
        


        differ_v = np.sum(v_d - v_s)
        cumulative_v_differ += differ_v
        log_v_differ.append(differ_v) 
        differ_exch = np.sum(exch_d - exch_s)
        cumulative_exch_differ += differ_exch
        log_exch_differ.append(differ_exch) 
        differ_har = np.sum(har_d - har_s)
        cumulative_har_differ += differ_har
        log_har_differ.append(differ_har) 
        differ_one = np.sum(one_d - one_s)
        cumulative_one_differ += differ_one
        log_one_differ.append(differ_one) 
        
        p.write(f"{j}\t{differ_v.real:.6f}\t{cumulative_v_differ.real:.6f}\t{differ_exch.real:.6f}\t{cumulative_exch_differ.real:.6f}\t{differ_har.real:.6f}\t{cumulative_har_differ.real:.6f}\t{differ_one.real:.6f}\t{cumulative_one_differ.real:.6f}\n")



    if True:#abs(energy.imag)<abs(trsh_imag) and abs(energy.real-e_hf.real)<abs(trsh_real): #ratio*(energy.real+en_const))):# and (abs((energy.real-e_hf.real)/(en_const+e_hf.real))<MAX_ACC_VAL):
        e_hf_d = e_hf_d *j +energy_new_d
        e_hf_d = e_hf_d /(j+1)
        e_hf_s = e_hf_s *j +energy_new_s
        e_hf_s = e_hf_s /(j+1)

        if first_cpu:
            print(j*D_TAU, e_hf_d, e_hf_s)#, e_one_new, hartree_new, exchange_new)
            txt = str(j*D_TAU) + '\t' + str(energy_new_d.real) + '\t' + str(energy_new_s.real) + '\t' + str(v_d.real) + '\t' + str(v_s.real) +'\t' + str(exch_d.real) + '\t' + str(exch_s.real) + '\t' + str(har_d.real) + '\t' + str(har_s.real) + '\t' + str(one_d.real) + '\t' + str(one_d.real) + '\t' + str(one_s.real) + '\n'
            file_out.write(txt)
            print()
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


# Check if the simulation has exceeded the maximum runtime (TMAX)
#    if time() - start_time > MAX_RUN_TIME:
#        print(f"Simulation stopped early at step {step} due to reaching TMAX.")
#        break


file_out.close()
    
if first_cpu:
    print('number of rare events = ',num_rare)
    print('total runtime = ', -start_time+time())
    print('Please see', output_file)
    print()
    print('##########################')
    print()

file_out = open(output_file,"r")
data1=np.loadtxt(file_out)
data = data1[:,[1]]
st = int(len(data)*EQUILIBRATION)
en = len(data)
a = blockAverage(data[st:en], block_divisor)
if first_cpu:
   print()
   print('Calculating ground state energy...')
   print()
   print ('E_gs_afqmc = ', np.mean(data[st:en]), '+-', np.sqrt(max(a[1])))
   print('Block mean = ', a[2])
   #print('Block size= ', a[0], 'variance= ', a[1])
file_out.close()
if first_cpu:
   print()


if first_cpu:
    with open('output.txt', 'a') as file_out:
        print(file=file_out)
        print('Calculating ground state energy...', file=file_out)
        print(file=file_out)
        print('E_gs_afqmc = ', np.mean(data[st:en]), '+-', np.sqrt(max(a[1])), file=file_out)
        print('total runtime = ', -start_time+time(), file=file_out)
 



#if first_cpu:
#   print()

#v, blockVar, blockMean = blockAverage(data[st:en])
#import matplotlib.pyplot as plt

#plt.figure(figsize=(14, 6))
#s= np.sqrt(blockVar)
## Plot block variance
#plt.subplot(1, 2, 1)
#plt.plot(v, s, label='Block Variance')
#plt.title('Block Variance vs. Block Size')
#plt.xlabel('Block Size')
#plt.ylabel('Variance')
#plt.legend()
#
## Plot block mean
#plt.subplot(1, 2, 2)
#plt.plot(v, blockMean, label='Block Mean', color='orange')
#plt.title('Block Mean vs. Block Size')
#plt.xlabel('Block Size')
#plt.ylabel('Mean')
#plt.legend()
#
#plt.tight_layout()
#plt.show()
