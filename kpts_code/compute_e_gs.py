import numpy as np

def blockAverage(datastream, isplot, maxBlockSize):
	Nobs         = len(datastream)           # total number of observations in datastream
	minBlockSize = 1;                        # min: 1 observation/block
 
	if maxBlockSize == 0:
		maxBlockSize = int(Nobs/4);        # max: 4 blocs (otherwise can't calc variance)
  
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

#data1=np.loadtxt('WALKERS128_AFQMC_CS_num_k64_C_H_Reortho_1_Rebal_5_TAU_0.00025.txt')
data1=np.loadtxt('1_AFQMC_CS_num_k64_C_H_Reortho_1_Rebal_10_TAU_0.0005.txt')

data = data1[:,[1]]
print(len(data))
st = int(len(data)*0.4)
en = int(len(data))

a = blockAverage(data[st:en:1], False, 0)
print()
print ('E_gs = ', np.mean(data[st:en:1]), '+-', np.sqrt(max(a[1])))
print()
#for i in range(len(a[1])):
#    print(np.sqrt(a[1][i]))
#print()
