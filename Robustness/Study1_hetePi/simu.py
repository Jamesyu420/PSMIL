import os 

NUM_CPU = len(os.sched_getaffinity(0))
print(f'Number of CPUs: {NUM_CPU}')
NUM_THREADS = 1
os.environ["MKL_NUM_THREADS"]     = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_NUM_THREADS"]     = str(NUM_THREADS)

NUM_PROCESS = int(NUM_CPU // NUM_THREADS) 
print(f'Number of PROCESSs: {NUM_PROCESS}')

from estimation import *
import numpy as np
import scipy.stats as stats
import time

# Data generating process with heterogeneous instance-level probability
def DataGen(mu1, mu0, Sigma, N, M, pis, lambda0, seed=0):
    p = len(mu1)
    np.random.seed(seed)
    Y = 1*(np.random.rand(N)<lambda0) 
    A = 1*(np.random.rand(N,M)<pis.reshape([-1,1]))
    A[Y == 0] =0 
    
    np.random.seed(seed)
    X1 = np.random.multivariate_normal(mu1, Sigma, np.sum(A))
    X0 = np.random.multivariate_normal(mu0, Sigma, np.sum(1-A))
    X = np.zeros([N*M, p]); 
    X[A.reshape(-1) == 1] = X1; 
    X[A.reshape(-1) == 0] = X0
    X = X.reshape([N,M,p])

    return {
        'pi': pis,
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
M = 1000
p = 50

np.random.seed(0)
rep = 500

# Obtain the empirical pi_i percentage
with open('/teams/WSIresult_1727165526/RealDataResult/512_75/train_XAY.pkl', 'rb') as file:
    data = pickle.load(file)
X = data['X'];Y = data['Y']; A = data['A']
pidata = np.mean(A[Y==1], axis = 1)
pi0 = np.mean(A[Y==1])

alphaN = 0

def simulation(r):
    
    prop = []
    ErrMu1_ins = []; ErrMu1_bag = []; ErrMu1_com = []
    ErrMu0_ins = []; ErrMu0_bag = []; ErrMu0_com = []
    ErrSigma_ins = []; ErrSigma_bag = []; ErrSigma_com = []
    ErrPi_ins = []; ErrPi_bag = []; ErrPi_com = []
    ErrPi_com = []; ErrPi_com = []; ErrPi_com = []
    Bagiter = []
    
    k = -1
    for N in Nlist:
        k += 1
        
        pis = np.random.choice(pidata, size=N, replace=True)
        data = DataGen(mu1, mu0, Sigma, N, M, pis, lambda0, seed=r+10)
        X, Y, A = data['X'], data['Y'], data['A']
              
        # Three estimators
        ins = InsMLE(X, A, Y)
        bag = BagMLE(X, Y, mu1, mu0, Sigma, pi_ground=pi0, iter=2000, tol=1e-6)
        Bagiter.append(bag['iter'])
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
              Bagiter)
    print(r)
    return results

import multiprocessing as mp
s1=time.time()
with mp.Pool(NUM_PROCESS) as pool:
    B = pool.map(simulation, range(rep))
s2=time.time()
print(s2-s1)

BB = np.array(B)

save_dir = "/teams/WSIresult_1727165526/RobustResult/Study1/"

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

np.savetxt(save_dir + "bag_iter.csv", BB[:,12,:], delimiter=',')

