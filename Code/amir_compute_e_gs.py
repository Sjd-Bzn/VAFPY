import numpy as np
import math

def blockAverage(datastream, isplot, maxBlockSize):
	Nobs         = len(datastream)           # total number of observations in datastream
	minBlockSize = 1;                        # min: 1 observation/block
 
	if maxBlockSize == 0:
		maxBlockSize = int(Nobs/15);        # max: 4 blocs (otherwise can't calc variance)
  
	NumBlocks = maxBlockSize - minBlockSize   # total number of block sizes

	blockMean = np.zeros(NumBlocks)               # mean (expect to be "nearly" constant)
	blockVar  = np.zeros(NumBlocks)               # variance associated with each blockSize
	blockCtr  = 0
	
				#
				#  blockSize is # observations/block
				#  run them through all the possibilities
				#
 
	for blockSize in range(minBlockSize, maxBlockSize):

		Nblock    = int(Nobs/blockSize)               # total number of such blocks in datastream
		obsProp   = np.zeros(Nblock)                  # container for parcelling block 

		# Loop to chop datastream into blocks
		# and take average
		for i in range(1,Nblock+1):
			
			ibeg = (i-1) * blockSize
			iend =  ibeg + blockSize
			obsProp[i-1] = np.mean(datastream[ibeg:iend])

		blockMean[blockCtr] = np.mean(obsProp)
		blockVar[blockCtr]  = (np.var(obsProp)/(Nblock - 1))
		blockCtr += 1
 
	v = np.arange(minBlockSize,maxBlockSize)
 
	if isplot:

		plt.subplot(2,1,1)
		plt.plot(v, np.sqrt(blockVar),'ro-',lw=2)
		plt.xlabel('block size')
		plt.ylabel(r'$\sigma$',fontsize=28)

		plt.subplot(2,1,2)
		plt.errorbar(v, blockMean, np.sqrt(blockVar))
		plt.ylabel(r'$E_{gs}^{hyb}$',fontsize=28)
		plt.xlabel('block size')

		print("E_gs = {0:f} +/- {1:f}\n".format(blockMean[-1], np.sqrt(blockVar[-1])))

		plt.tight_layout()
		plt.show()
                #plt.savefig('E_gs_block_avg_hyb_rebal1.eps',bbox_inches='tight')
		
	return v, blockVar, blockMean


da_1=np.loadtxt('numpy_double.txt')
#data1=np.loadtxt('up_s.txt')

da_2=np.loadtxt('numpy_single.txt')
da_3=np.loadtxt('jax_double.txt')
da_4=np.loadtxt('jax_single.txt')
#da_5=np.loadtxt('AFQMC_CS_num_k1_samp_freq_1_Reortho_1_Rebal_4_TAU_0.005_walkers_100.txt')
#da_6=np.loadtxt('AFQMC_CS_num_k1_samp_freq_1_Reortho_1_Rebal_2_TAU_0.007_walkers_100.txt')


data_1 = da_1[:,[1]]
tau = data_1[:,[0]]
#print(len(data))
#tot_d= da_d[:, [3]]
#weight = data_1[:,[-1]]
st = int(len(data_1)*0.25)
en = len(data_1)



a_1 = blockAverage(data_1[st:en:1], False, 0)
#b_d = blockAverage(tot_1[st:en:1], False, 0)
print('E_gs numpy double= ', np.mean(data_1[st:en:1]), '+-', np.sqrt(max(a_1[1])))
#print ('E_gs 2= ', np.mean(tot_d[st:en:1]), '+-', np.sqrt(max(b_d[1])))
#print('E_gs new1 = ', np.sum(tot_d[st:en:1])/np.sum(weight[st:en:1]), '+-', np.sqrt(max(b_d[1])))

data_2 = da_2[:,[1]]
#tau = data_2[:,[0]]
#print(len(data))
#tot_d= da_d[:, [3]]
#weight = data_1[:,[-1]]
st = int(len(data_2)*0.25)
en = len(data_2)



a_2 = blockAverage(data_2[st:en:1], False, 0)
#b_d = blockAverage(tot_1[st:en:1], False, 0)
print('E_gs numpy single= ', np.mean(data_2[st:en:1]), '+-', np.sqrt(max(a_2[1])))

data_3 = da_3[:,[1]]
#tau = data_3[:,[0]]
#print(len(data))
#tot_d= da_d[:, [3]]
#weight = data_1[:,[-1]]
st = int(len(data_3)*0.25)
en = len(data_3)



a_3 = blockAverage(data_3[st:en:1], False, 0)
#b_d = blockAverage(tot_1[st:en:1], False, 0)
print('E_gs jax double= ', np.mean(data_3[st:en:1]), '+-', np.sqrt(max(a_3[1])))
# Sample data

data_4 = da_4[:,[1]]
#tau = data_4[:,[0]]
#print(len(data))
#tot_d= da_d[:, [3]]
#weight = data_1[:,[-1]]
st = int(len(data_4)*0.25)
en = len(data_4)



a_4 = blockAverage(data_4[st:en:1], False, 0)
#b_d = blockAverage(tot_1[st:en:1], False, 0)
print('E_gs jax single= ', np.mean(data_4[st:en:1]), '+-', np.sqrt(max(a_4[1])))

#data_5 = da_5[:,[1]]
##tau = data_5[:,[0]]
##print(len(data))
##tot_d= da_d[:, [3]]
##weight = data_1[:,[-1]]
#st = int(len(data_5)*0.25)
#en = len(data_5)
#
#
#
#a_5 = blockAverage(data_5[st:en:1], False, 0)
##b_d = blockAverage(tot_1[st:en:1], False, 0)
#print('E_gs 0.005= ', np.mean(data_5[st:en:1]), '+-', np.sqrt(max(a_5[1])))
## Calculate standard deviation
#
#data_6 = da_6[:,[1]]
##tau = data_5[:,[0]]
##print(len(data))
##tot_d= da_d[:, [3]]
##weight = data_1[:,[-1]]
#st = int(len(data_6)*0.25)
#en = len(data_6)
#
#
#
#a_6 = blockAverage(data_6[st:en:1], False, 0)
##b_d = blockAverage(tot_1[st:en:1], False, 0)
#print('E_gs 0.007= ', np.mean(data_6[st:en:1]), '+-', np.sqrt(max(a_6[1])))
##std_dev = np.std(data_d[st:en:1], ddof=1) 
##std_error = std_dev / math.sqrt(en)
##print(f"Standard Error: {std_error}")
#
##v, blockVar, blockMean = blockAverage(data_d[st:en], False, 0)
##import matplotlib.pyplot as plt
##
##plt.figure(figsize=(14, 6))
##s= np.sqrt(blockVar)
### Plot block variance
##plt.subplot(1, 2, 1)
##plt.plot(v, s, label='Block Variance')
##plt.title('Block Variance vs. Block Size Update Double')
##plt.xlabel('Block Size')
##plt.ylabel('Variance')
##plt.legend()
##
### Plot block mean
#plt.subplot(1, 2, 2)
#plt.plot(tau, data_d, label='Block Mean', color='orange')
#plt.title('Block Mean vs. Block Size Update Double')
#plt.xlabel('Block Size')
#plt.ylabel('Mean')
#plt.legend()
#
#plt.tight_layout()
#plt.show()



#data1=np.loadtxt('AFQMC_CS_num_k1_samp_freq_1_Reortho_1_Rebal_10_TAU_0.005_walkers_52.txt')
#data1=np.loadtxt('up_s.txt')
#
#
#
#data = data1[:,[1]]
#tau = data1[:,[0]]
##print(len(data))
#tot= data1[:, [2]]
##weight = data1[:,[-1]]
#st = int(len(data)*0.25)
#en = len(data)
#
#
#
#a = blockAverage(data[st:en:1], False, 0)
#b = blockAverage(tot[st:en:1], False, 0)
#print('E_gs double update single= ', np.mean(data[st:en:1]), '+-', np.sqrt(max(a[1])))
#print ('E_gs single update single= ', np.mean(tot[st:en:1]), '+-', np.sqrt(max(b[1])))
##print('E_gs new1 = ', np.sum(tot[st:en:1])/np.sum(weight[st:en:1]), '+-', np.sqrt(max(a[1])))
#
#
## Sample data
#
#
## Calculate standard deviation
#
#std_dev = np.std(data[st:en:1], ddof=1) 
#std_error = std_dev / math.sqrt(en)
#print(f"Standard Error: {std_error}")
#
#v, blockVar, blockMean = blockAverage(data[st:en], False, 0)
#import matplotlib.pyplot as plt
#
#plt.figure(figsize=(14, 6))
#s= np.sqrt(blockVar)
## Plot block variance
#plt.subplot(1, 2, 1)
#plt.plot(v, s, label='Block Variance')
#plt.title('Block Variance vs. Block Size update single')
#plt.xlabel('Block Size')
#plt.ylabel('Variance')
#plt.legend()
#
## Plot block mean
#plt.subplot(1, 2, 2)
#plt.plot(tau, data, label='Block Mean', color='orange')
#plt.title('Block Mean vs. Block Size')
#plt.xlabel('Block Size')
#plt.ylabel('Mean')
#plt.legend()
#
#plt.tight_layout()
#plt.show()
