#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
def block_average(data, max_block_size=None):
    """
    Perform standard block averaging (reblocking) on a 1D data set to estimate
    the standard error of the mean, accounting for correlations.

    Parameters
    ----------
    data : array-like
        1D array of scalar data points (e.g. correlated MC measurements).
    max_block_size : int, optional
        Largest block size to test. If None, defaults to N//2, where N = len(data).

    Returns
    -------
    block_sizes : np.ndarray
        Array of block sizes (from 1 up to max_block_size).
    var_of_block_means : np.ndarray
        Sample variance of the block means for each block size.
    standard_error : np.ndarray
        Estimated standard error of the mean for each block size,
        computed as sqrt( var_of_block_means / (Nblocks) ).
    """
    data = np.array(data, copy=False)
    N = len(data)

    if max_block_size is None:
        max_block_size = N // 2  # A reasonable default

    # Arrays to store results
    block_sizes = np.arange(1, max_block_size + 1)
    var_of_block_means = np.zeros_like(block_sizes, dtype=float)
    standard_error = np.zeros_like(block_sizes, dtype=float)

    for idx, b in enumerate(block_sizes):
        # Number of blocks for this block size
        Nblocks = N // b

        # Compute block means
        block_means = np.zeros(Nblocks, dtype=float)
        for j in range(Nblocks):
            block_means[j] = np.mean(data[j*b : (j+1)*b])

        # Unbiased sample variance of the block means
        vm = np.var(block_means, ddof=1)  # divides by Nblocks - 1
        var_of_block_means[idx] = vm

        # Standard error of the overall mean
        standard_error[idx] = np.sqrt(vm / Nblocks)

    return block_sizes, var_of_block_means, standard_error


def main():
    # --------------------------------------------------
    # 1) Load data from a file with 4 columns
    #    (Replace 'my_data.txt' with your actual filename)
    # --------------------------------------------------
    #filenames = ["numpy_double.txt","numpy_single.txt","numpy_single_1.txt","numpy_single_2.txt","jax_double.txt","jax_single.txt","jax_single_1.txt","jax_single_2.txt","jax_single_3.txt","jax_single_4.txt"]
    filenames = ["AFQMC_CS_num_k1_samp_freq_1_Reortho_1_Rebal_20_TAU_0.0005_walkers_156.txt"]
    for filename in filenames:
        data_all = np.loadtxt(filename)  # shape: (num_rows, 4)

         # --------------------------------------------------
         # 2) Select the second column, i.e. data_all[:,1]
         # --------------------------------------------------
        column_index = 1  # 0-based; 1 means "2nd column"
        data_full = data_all[:, column_index]
        time_index = 0
        time_full = data_all[:, time_index]
     # --------------------------------------------------
     # 3) Discard first 25% as burn-in
     # --------------------------------------------------
        st = int(0.25 * len(data_full))
        data = data_full[st:]
        time = time_full[st:]
     # --------------------------------------------------
     # 4) Perform block averaging
     # --------------------------------------------------
        block_sizes, varBM, se = block_average(data, max_block_size=1000)

     # --------------------------------------------------
     # 5) Compare with naive standard error
     # --------------------------------------------------
        N_prod = len(data)
        naive_std = np.std(data, ddof=1)         # sample stdev
        naive_se  = naive_std / np.sqrt(N_prod)  # naive standard error

     # --------------------------------------------------
     # 6) Plot results
     # --------------------------------------------------
        plt.figure(figsize=(20,10))

     # (a) Plot sqrt(var_of_block_means) = std of block means
        plt.subplot(1,2,1)
        plt.plot(block_sizes, np.sqrt(varBM), 'o-', label='Std of Block Means')
        plt.xlabel('Block Size')
        plt.ylabel('Std of Block Means')
        plt.title('Block Means Fluctuation')
        plt.legend()

        # (b) Standard Error vs block size
        plt.subplot(1,2,2)
        plt.plot(block_sizes, se, 'o-', label='Block-Averaged SE')
        plt.axhline(naive_se, color='r', linestyle='--',
                    label=f'Naive SE = {naive_se:.5f}')
        plt.xlabel('Block Size')
        plt.ylabel('Std Error of the Mean')
        plt.title('Standard Error vs. Block Size')
        plt.legend()
        
        # (c) Data vs tau
        plt.subplot(2,2,1)
        plt.plot(time, data, 'o-', label='Raw Data')
        plt.xlabel('Time')
        plt.ylabel('Raw Energy in each Step')
        plt.legend()
        pdf_filename = os.path.splitext(filename)[0] + "_plot.pdf"
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        plt.tight_layout()
        plt.show()

         # --------------------------------------------------
         # 7) Choose final error estimate from largest block size
         #    or from a plateau region. Here we pick the largest.
         # --------------------------------------------------
        system_name = os.path.splitext(filename)[0]
        final_idx = -1  # use the last element in 'se'
        final_bsize = block_sizes[final_idx]
        final_err   = se[final_idx]
        mean_val    = np.mean(data)

         # --------------------------------------------------
         # 8) Print final results
         # --------------------------------------------------
        
        print(f"System = {system_name}")
        print(f"Data column used = {column_index+1}")
        print(f"Production data length = {N_prod}")
        print(f"Final chosen block size = {final_bsize}")
        print(f"Mean value     = {mean_val:.6f}")
        print(f"Std. Error     = {final_err:.6f}")
        print(f"Naive SE       = {naive_se:.6f}")

if __name__ == '__main__':
    main()

