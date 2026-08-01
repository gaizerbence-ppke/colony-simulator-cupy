import cupy as cp

def initialize_binomial_distribution_matrix(N):
    binomialDistributionMatrix = cp.zeros((N, N), dtype=cp.float32)
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