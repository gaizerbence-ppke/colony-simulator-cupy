import numpy as np
import scipy.fft

def setup_array_backend():
    try:
        import cupy as cp
        import cupyx.scipy.fft as cufft
        
        if cp.is_available():
            print("Cuda available")
            return cp, cufft
        else:
            print("Cuda NOT available, fallback to CPU")
            return np, scipy.fft
    except (ImportError, Exception):
        print("Cuda NOT available, fallback to CPU")
        return np, scipy.fft

xp, xfft = setup_array_backend()

def initialize_binomial_distribution_matrix(N):
    binomialDistributionMatrix = xp.zeros((N, N), dtype=xp.float32)
    binomialDistributionMatrix[0, 0] = 1.0
    for n in range(1, N):
        for k in range(n + 1):
            if k == 0 or k == n:
                binomialDistributionMatrix[n, k] = 1.0
            else:
                binomialDistributionMatrix[n, k] = binomialDistributionMatrix[n-1, k-1] + binomialDistributionMatrix[n-1, k]
    for n in range(N):
        binomialDistributionMatrix[n, :n+1] /= 2 ** n
    return binomialDistributionMatrix