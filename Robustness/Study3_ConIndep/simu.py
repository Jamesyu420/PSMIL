import os 

NUM_CPU = len(os.sched_getaffinity(0))
print(f'Number of CPUs: {NUM_CPU}')
NUM_THREADS = 1
os.environ["MKL_NUM_THREADS"]     = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_NUM_THREADS"]     = str(NUM_THREADS)

NUM_PROCESS = int(NUM_CPU // NUM_THREADS) # 可用在mp.Pool(NUM_PROCESS)中
print(f'Number of PROCESSs: {NUM_PROCESS}')

from estimation import *
import numpy as np
import scipy.stats as stats
import time

# Data generating process
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


import pickle
# Real data based true parameter
with open('../para_est_KmeansInsInit.pkl', 'rb') as file:
    data = pickle.load(file)
mu1 = data['ins']['mu1'][:50]
mu0 = data['ins']['mu0'][:50]
Sigma = data['ins']['Sigma'][:50,:50]
pi0 = data['ins']['pi']
lambda0 = data['ins']['alpha']
Omega = np.linalg.inv(Sigma)

Nlist =  np.array([50, 100, 200, 500, 1000])
M = 100                                       # M not too large
p = 50

np.random.seed(0)
rep = 500

alphaN = 0

def simulation(r):
    
    prop = []
    ErrMu1_ins = []; ErrMu1_bag = []; ErrMu1_com = []
    ErrMu0_ins = []; ErrMu0_bag = []; ErrMu0_com = []
    ErrSigma_ins = []; ErrSigma_bag = []; ErrSigma_com = []
    ErrPi_ins = []; ErrPi_bag = []; ErrPi_com = []
    ErrPi_com = []; ErrPi_com = []; ErrPi_com = []
    Violated = []
    
    k = -1
    for N in Nlist: 
        k += 1
        
        data = DataGen(mu1, mu0, Sigma, N, M, lambda0, pi0, seed=r+10)
        X, Y, A = data['X'], data['Y'], data['A']
        Violated.append(np.sum((np.sum(A,axis=1)>0) != Y))                      # When Aim=0 but Y=1
        
        # Three estimators
        ins = InsMLE(X, A, Y)
        bag = BagMLE(X, Y, mu1, mu0, Sigma, pi_ground=pi0, iter=2000, tol=1e-6)
        com = SubMLE(X, Y, A, mu1, mu0, Sigma, pi0, bag, 
                     alpha_n=alphaN, iter=1000, tol=1e-6, seed=0)
        
        mu1_true = np.array(mu1.copy()); mu0_true = np.array(mu0.copy())
        
        # MSE
        ErrMu1_ins.append(L2norm(mu1_true, ins["mu1"]) / p)
        ErrMu1_bag.append(L2norm(mu1_true, np.array(bag["mu1"])) / p)
        ErrMu1_com.append(L2norm(mu1_true, np.array(com["mu1"])) / p)
        
        ErrMu0_ins.append(L2norm(mu0_true, ins["mu0"]) / p)
        ErrMu0_bag.append(L2norm(mu0_true, np.array(bag["mu0"])) / p)
        ErrMu0_com.append(L2norm(mu0_true, np.array(com["mu0"])) / p)

        ErrSigma_ins.append(np.linalg.norm(Omega - np.linalg.inv(ins["Sigma"]),ord ='fro')**2 / p**2)
        ErrSigma_bag.append(np.linalg.norm(Omega - np.linalg.inv(bag["Sigma"]),ord ='fro')**2 / p**2)
        ErrSigma_com.append(np.linalg.norm(Omega - np.linalg.inv(com["Sigma"]),ord ='fro')**2 / p**2)

        ErrPi_ins.append(L2norm(pi0, ins["pi"]))
        ErrPi_bag.append(L2norm(pi0, np.array(bag["pi"])))
        ErrPi_com.append(L2norm(pi0, np.array(com["pi"])))
        
    results = (ErrMu1_ins,ErrMu1_bag, ErrMu1_com,
              ErrMu0_ins,ErrMu0_bag, ErrMu0_com,
              ErrSigma_ins,ErrSigma_bag, ErrSigma_com,
              ErrPi_ins,ErrPi_bag, ErrPi_com,
              Violated)
    print(r)
    return results

import multiprocessing as mp
s1=time.time()
with mp.Pool(NUM_PROCESS) as pool:
    B = pool.map(simulation, range(rep))
s2=time.time()
print(s2-s1)

BB = np.array(B)

save_dir = "/teams/WSIresult_1727165526/RobustResult/Study3/"

np.savetxt(save_dir + "mu1_ins.csv", BB[:,0,:], delimiter=',')
np.savetxt(save_dir + "mu1_bag.csv", BB[:,1,:], delimiter=',')
np.savetxt(save_dir + "mu1_com.csv", BB[:,2,:], delimiter=',')

np.savetxt(save_dir + "mu0_ins.csv", BB[:,3,:], delimiter=',')
np.savetxt(save_dir + "mu0_bag.csv", BB[:,4,:], delimiter=',')
np.savetxt(save_dir + "mu0_com.csv", BB[:,5,:], delimiter=',')

np.savetxt(save_dir + "Sigma_ins.csv", BB[:,6,:], delimiter=',')
np.savetxt(save_dir + "Sigma_bag.csv", BB[:,7,:], delimiter=',')
np.savetxt(save_dir + "Sigma_com.csv", BB[:,8,:], delimiter=',')

np.savetxt(save_dir + "pi_ins.csv", BB[:,9,:], delimiter=',')
np.savetxt(save_dir + "pi_bag.csv", BB[:,10,:], delimiter=',')
np.savetxt(save_dir + "pi_com.csv", BB[:,11,:], delimiter=',')

np.savetxt(save_dir + "violated.csv", BB[:,12,:], delimiter=',')

