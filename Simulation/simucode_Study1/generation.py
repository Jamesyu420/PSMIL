import numpy as np
import scipy.stats as stats

def DataGen(mu1, mu0, Sigma, N, M, lambda0=0.5, pi0=0.05, seed=0):
    np.random.seed(seed)
    p = len(mu1)
    X = np.zeros((N, M, p))
    A = np.zeros((N, M))
    Y = stats.bernoulli.rvs(lambda0, size=N)

    for i in range(N):
        if Y[i] == 0:
            A[i, :] = 0
        else:
            A[i, :] = stats.bernoulli.rvs(pi0, size=M)

    X1 = np.random.multivariate_normal(mu1, Sigma, N*M).reshape([N,M,p])
    X0 = np.random.multivariate_normal(mu0, Sigma, N*M).reshape([N,M,p])
    A_expanded = A[:, :, np.newaxis]              # Expanding A to be broadcastable
    X = A_expanded * X1 + (1 - A_expanded) * X0

    return {
        'mu1': mu1,
        'mu0': mu0,
        'Sigma': Sigma,
        'X': X,
        'A': A,
        'Y': Y
    }
